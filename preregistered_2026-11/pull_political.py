"""Pull settled Kalshi Elections/Politics markets and reconstruct the entry book exactly as in the
frozen retrospective stratum (earliest daily candle with a two-sided book). Swan (2026).
Pre-specified in preregistered_2026-11/PLAN_political_compression_test.md (commit 8e46d8e).
Usage: python pull_political.py SERIES_JSON OUT_DIR"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_JSON, OUT = sys.argv[1], sys.argv[2]
LO, HI = datetime.fromisoformat("2025-08-01T00:00:00+00:00").timestamp(), datetime.fromisoformat("2026-11-04T23:59:59+00:00").timestamp()
lock = threading.Lock(); last = [0.0]; RATE = 1 / 12
def get(path):
    for attempt in range(6):
        with lock:
            wait = last[0] + RATE - time.time()
            if wait > 0: time.sleep(wait)
            last[0] = time.time()
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "academic-research"})
            with urllib.request.urlopen(req, timeout=40) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500: time.sleep(2 ** attempt); continue
            return None
        except Exception: time.sleep(2 ** attempt)
    return None
def ts(s):
    try: return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception: return None
def px(d):
    if not isinstance(d, dict): return None
    if d.get("close") is not None: return float(d["close"])
    if d.get("close_dollars") not in (None, ""): return round(float(d["close_dollars"]) * 100, 2)
    return None
series = [s for s in json.load(open(SERIES_JSON))["series"] if s.get("category") in ("Elections", "Politics")]
os.makedirs(OUT, exist_ok=True)
mk_path, done_path = os.path.join(OUT, "markets.jsonl"), os.path.join(OUT, "entries.jsonl")
# 1) list settled markets per series (resumable)
if not os.path.exists(mk_path):
    def list_series(s):
        out, cur = [], ""
        while True:
            d = get(f"/markets?series_ticker={s['ticker']}&status=settled&limit=1000" + (f"&cursor={cur}" if cur else ""))
            if not d: break
            for m in d.get("markets", []):
                o, c = ts(m.get("open_time")), ts(m.get("close_time"))
                if m.get("result") in ("yes", "no") and m.get("market_type", "binary") == "binary" and o and c and LO <= c <= HI and c - o >= 6 * 3600:
                    out.append({"ticker": m["ticker"], "event": m.get("event_ticker"), "series": s["ticker"], "category": s.get("category"),
                                "open": o, "close": c, "result": m["result"]})
            cur = d.get("cursor") or ""
            if not cur: break
        return out
    allm = []
    with ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(list_series, series)):
            allm.extend(res)
            if i % 500 == 0: print(f"series {i}/{len(series)} markets so far {len(allm)}", flush=True)
    with open(mk_path, "w") as f:
        for m in allm: f.write(json.dumps(m) + "\n")
markets = [json.loads(l) for l in open(mk_path)]
print("eligible settled political markets:", len(markets), "| events:", len({m['event'] for m in markets}), flush=True)
# 2) entry book from daily candles (resumable)
done = set()
if os.path.exists(done_path):
    for l in open(done_path):
        try: done.add(json.loads(l)["ticker"])
        except Exception: pass
todo = [m for m in markets if m["ticker"] not in done]
wlock = threading.Lock()
def entry(m):
    d = get(f"/series/{m['series']}/markets/{m['ticker']}/candlesticks?start_ts={int(m['open'])}&end_ts={int(m['close'])}&period_interval=1440")
    rec = {**m, "b": None, "a": None, "trade": None, "entry_ts": None, "status": "no candles"}
    for c in (d or {}).get("candlesticks", []):
        b, a = px(c.get("yes_bid")), px(c.get("yes_ask"))
        if b is not None and a is not None and b > 0 and a < 100 and a > b:
            vol = c.get("volume") if c.get("volume") is not None else float(c.get("volume_fp") or 0)
            rec.update({"b": b, "a": a, "trade": px(c.get("price")) if (vol or 0) > 0 else None, "entry_ts": c.get("end_period_ts"), "status": "ok"})
            break
    else:
        if d and d.get("candlesticks"): rec["status"] = "no two-sided book"
    with wlock:
        with open(done_path, "a") as f: f.write(json.dumps(rec) + "\n")
with ThreadPoolExecutor(6) as ex:
    for i, _ in enumerate(ex.map(entry, todo)):
        if i % 2000 == 0: print(f"candles {i}/{len(todo)}", flush=True)
rows = [json.loads(l) for l in open(done_path)]
print("done. entries:", len(rows), "| two-sided:", sum(r["status"] == "ok" for r in rows), flush=True)
