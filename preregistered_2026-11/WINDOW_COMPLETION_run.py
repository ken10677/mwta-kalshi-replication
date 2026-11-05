"""Window completion (run AFTER the pre-registered results; not pre-registered).
Applies the pre-registered analysis (analyze_political.py, unchanged) to
  (a) the markets the pre-registered pull missed (settled before Kalshi's historical cutoff), and
  (b) the full window the plan named (live + historical, de-duplicated by ticker).
Usage: python WINDOW_COMPLETION_run.py LIVE_ENTRIES HIST_ENTRIES OUT_DIR"""
import json, os, subprocess, sys
LIVE, HIST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
live = [json.loads(l) for l in open(LIVE)]
seen = {r["ticker"] for r in live}
hist = [r for r in (json.loads(l) for l in open(HIST)) if r["ticker"] not in seen]
os.makedirs(OUT, exist_ok=True)
paths = {"historical_only": os.path.join(OUT, "entries_historical_only.jsonl"), "full_window": os.path.join(OUT, "entries_full_window.jsonl")}
with open(paths["historical_only"], "w") as f:
    for r in hist: f.write(json.dumps(r) + "\n")
with open(paths["full_window"], "w") as f:
    for r in live + hist: f.write(json.dumps(r) + "\n")
summary = {"live_markets": len(live), "historical_markets_new": len(hist),
           "historical_two_sided": sum(r["status"] == "ok" for r in hist)}
for k, p in paths.items():
    out = os.path.join(OUT, f"RESULTS_{k}.json")
    subprocess.run([sys.executable, "-I", os.path.join(here, "analyze_political.py"), p, out], check=True, capture_output=True)
    summary[k] = json.load(open(out))
json.dump(summary, open(os.path.join(OUT, "WINDOW_COMPLETION_summary.json"), "w"), indent=1)
for k in paths:
    R = summary[k]
    print("==", k, R["universe"], "|", R["verdict_preregistered"])
    for kk, v in R["midpoint"].items(): print(f"  {kk:<32} n={v['n']:>6} ev={v['events']:>5} slope {v['slope']} {v['slope_ci']} int {v['intercept']} gap {v['gap']} {v['gap_ci']}")
    for kk, v in R["trade_price"].items(): print(f"  trade {kk:<26} n={v['n']:>6} slope {v['slope']} {v['slope_ci']} gap {v['gap']} {v['gap_ci']}")
