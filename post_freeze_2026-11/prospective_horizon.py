#!/usr/bin/env python3
"""Prospective two-sided stratum: entry horizon and book width (post-freeze, exploratory). Swan (2026).
Usage: python prospective_horizon.py [LOGS_DIR] [OUT_DIR]"""
import json, re, sys
import numpy as np, pandas as pd
LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rng = np.random.default_rng(20260824)
res = {}
for line in open(f"{LOGS}/resolutions.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") in ("yes", "no"): res[r["t"]] = r["r"]
first = {}
for line in open(f"{LOGS}/market_snapshots.jsonl", encoding="utf-8"):
    s = json.loads(line)
    if s["t"] not in first: first[s["t"]] = s
TH = re.compile(r"-[TB][\d.\-]+$")
pr = pd.DataFrame([{"t": t, "event": TH.sub("", t), "b": s.get("b") or 0, "a": s.get("a") or 100, "p": s["p"],
                    "htc": s.get("htc"), "y": 1 if res[t] == "yes" else 0} for t, s in first.items() if t in res])
pr = pr[(pr.b > 0) & (pr.a < 100) & (pr.a > pr.b)].copy(); assert len(pr) == 583
pr["mid"] = (pr.a + pr.b) / 2; pr["spread"] = pr.a - pr.b; pr["gap"] = pr.y * 100 - pr.mid
def boot(sub, draws=10000):
    g = sub.groupby("event")["gap"].agg(["sum", "count"]); S, N = g["sum"].to_numpy(float), g["count"].to_numpy(); k = len(S)
    if k == 0: return None
    out = np.empty(draws)
    for i in range(0, draws, 500):
        m = min(500, draws - i); idx = rng.integers(0, k, size=(m, k)); out[i:i+m] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"est": round(float(S.sum()/N.sum()), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(len(sub)), "clusters": int(k)}
h = pr.htc.astype(float)
R = {"median_htc_hours": round(float(h.median()), 1), "share_ge_7d": round(float((h >= 168).mean()), 3),
     "share_lt_24h": round(float((h < 24).mean()), 3), "median_spread": float(pr.spread.median()), "strata": {}}
for name, lo, hi in (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 1e12)):
    s = pr[(h >= lo) & (h < hi)]
    R["strata"][name] = {"all": boot(s), "spread<=50": boot(s[s.spread <= 50])}
print(json.dumps(R, indent=1)); json.dump(R, open(f"{OUT}/prospective_horizon.json", "w"), indent=1)
