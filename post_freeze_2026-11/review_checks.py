#!/usr/bin/env python3
"""Checks prompted by pre-submission review (post-freeze, exploratory). Swan (2026).
Calibration slope intervals; entries near close (possible in-play quotes); partition vs ladder series;
further series-family drops; prospective gap by event size.
Usage: python review_checks.py [LOGS_DIR] [OUT_DIR]"""
import json, re, sys
from datetime import datetime
import numpy as np, pandas as pd
LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rng = np.random.default_rng(20260824)
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"): continue
    tth = None
    try:
        cd = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
        if r.get("qts"): tth = (cd.timestamp() - float(r["qts"])) / 3600
    except Exception: pass
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "",
                 "b": int(r.get("b") or 0), "a": int(r.get("a") or 100), "y": 1 if r["r"] == "yes" else 0, "tth": tth})
df = pd.DataFrame(rows); df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy(); assert len(df) == 57808
df["mid"] = (df.a + df.b) / 2; df["spread"] = df.a - df.b; df["gap"] = df.y * 100 - df.mid
df["ev_n"] = df.groupby("event")["t"].transform("count")
SEG = (df.spread >= 51) & (df.ev_n >= 3)
def boot(sub, draws=10000):
    g = sub.groupby("event")["gap"].agg(["sum", "count"]); S, N = g["sum"].to_numpy(float), g["count"].to_numpy(); k = len(S)
    if k == 0: return None
    out = np.empty(draws)
    for i in range(0, draws, 500):
        m = min(500, draws - i); idx = rng.integers(0, k, size=(m, k)); out[i:i+m] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"est": round(float(S.sum()/N.sum()), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(len(sub)), "clusters": int(k)}
def slope_boot(sub, draws=2000):
    x = sub.mid / 100; y = sub.y.astype(float)
    g = pd.DataFrame({"e": sub.event, "n": 1, "x": x, "y": y, "xx": x * x, "xy": x * y}).groupby("e").sum().to_numpy()
    def sl(t):
        n, sx, sy, sxx, sxy = t[..., 0], t[..., 1], t[..., 2], t[..., 3], t[..., 4]
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)
    pt = sl(g.sum(0)); k = len(g); out = np.empty(draws)
    for i in range(0, draws, 200):
        m = min(200, draws - i); idx = rng.integers(0, k, size=(m, k)); out[i:i+m] = sl(g[idx].sum(1))
    lo, hi = np.percentile(out, [2.5, 97.5]); return {"slope": round(float(pt), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}
R = {}
R["slope"] = {"all": slope_boot(df), "spread<=50": slope_boot(df[df.spread <= 50]), "all_except_segment": slope_boot(df[~SEG])}
near = df.tth.notna() & (df.tth < 3)
s = df[SEG]
R["near_close"] = {"segment_n": int(len(s)), "segment_tth_lt3h_n": int((near & SEG).sum()),
    "segment_tth_lt3h": boot(df[SEG & near]), "segment_tth_ge3h": boot(df[SEG & ~near]),
    "segment_tth_p5_p50": [round(float(s.tth.quantile(.05)), 1), round(float(s.tth.quantile(.5)), 1)],
    "top_series_lt3h": df[SEG & near].series.value_counts().head(6).to_dict(),
    "all_tth_lt3h_n": int(near.sum()), "all_tth_lt3h_share": round(float(near.mean()), 3)}
d3 = df[~near].copy(); d3["ev_n"] = d3.groupby("event")["t"].transform("count")  # recount within the subsample
seg3 = (d3.spread >= 51) & (d3.ev_n >= 3)
R["excl_lt3h"] = {"n": int(len(d3)), "all": boot(d3), "spread<=50": boot(d3[d3.spread <= 50]),
    "segment": boot(d3[seg3]), "segment_share": round(float(seg3.mean()), 3), "all_except_segment": boot(d3[~seg3]),
    "wide_1-2": boot(d3[(d3.spread >= 51) & (d3.ev_n <= 2)])}
R["excl_lt3h"]["segment_contribution"] = round(R["excl_lt3h"]["segment_share"] * R["excl_lt3h"]["segment"]["est"], 2)
part = s.series == "KXCLUBFGAME"
ladder = s.series.isin(["KXCLUBFTOTAL", "KXCLUBFSPREAD"])
R["segment_by_type"] = {"soccer_match_result": {**boot(df[SEG & (df.series == "KXCLUBFGAME")]), "yes": round(float(100*s[part].y.mean()), 1), "mid": round(float(s[part].mid.mean()), 1)},
    "soccer_total_spread_ladders": {**boot(df[SEG & df.series.isin(["KXCLUBFTOTAL", "KXCLUBFSPREAD"])]), "yes": round(float(100*s[ladder].y.mean()), 1), "mid": round(float(s[ladder].mid.mean()), 1)},
    "other": {**boot(df[SEG & ~df.series.str.startswith("KXCLUBF")]), "yes": round(float(100*s[~s.series.str.startswith("KXCLUBF")].y.mean()), 1)}}
R["drop_more"] = []
for name, mask in (("WNBA", df.series.str.startswith("KXWNBA")), ("NBA summer league", df.series.str.startswith("KXNBASUMMER")),
                   ("all basketball (KXWNBA, KXNBA)", df.series.str.startswith("KXWNBA") | df.series.str.startswith("KXNBA"))):
    x = df[~mask]; R["drop_more"].append({"dropped": name, "n_dropped": int(mask.sum()), "n_dropped_in_segment": int((mask & SEG).sum()),
        "narrow": boot(x[x.spread <= 50]), "wide_3+": boot(x[(x.spread >= 51) & (x.ev_n >= 3)])})
# prospective by event size (two-sided, earliest snapshot; same loader as gate_analysis.load_pairs)
res = {}
for line in open(f"{LOGS}/resolutions.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") in ("yes", "no"): res[r["t"]] = r["r"]
first = {}
for line in open(f"{LOGS}/market_snapshots.jsonl", encoding="utf-8"):
    q = json.loads(line)
    if q["t"] not in first: first[q["t"]] = q
TH = re.compile(r"-[TB][\d.\-]+$")
pr = pd.DataFrame([{"t": t, "event": TH.sub("", t), "b": q.get("b") or 0, "a": q.get("a") or 100, "y": 1 if res[t] == "yes" else 0}
                   for t, q in first.items() if t in res])
pr = pr[(pr.b > 0) & (pr.a < 100) & (pr.a > pr.b)].copy(); pr["mid"] = (pr.a + pr.b) / 2; pr["gap"] = pr.y * 100 - pr.mid
pr["spread"] = pr.a - pr.b; pr["ev_n"] = pr.groupby("event")["t"].transform("count")
R["prospective_by_eventsize"] = {"1-2": boot(pr[pr.ev_n <= 2]), "3+": boot(pr[pr.ev_n >= 3]),
                                 "3+_narrow": boot(pr[(pr.ev_n >= 3) & (pr.spread <= 50)])}
print(json.dumps(R, indent=1)); json.dump(R, open(f"{OUT}/review_checks.json", "w"), indent=1)
