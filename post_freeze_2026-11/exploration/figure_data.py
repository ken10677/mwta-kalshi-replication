#!/usr/bin/env python3
"""Reliability-curve data (Figure 1) and mechanism numbers. Swan (2026), post-freeze, exploratory.
Usage: python figure_data.py [LOGS_DIR] [OUT_DIR]"""
import json, sys
import numpy as np, pandas as pd
LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rng = np.random.default_rng(20260824)
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"): continue
    b, a = int(r.get("b") or 0), int(r.get("a") or 100)
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "b": b, "a": a, "y": 1 if r["r"] == "yes" else 0})
df = pd.DataFrame(rows); df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy()
df["mid"] = (df.a + df.b) / 2; df["spread"] = df.a - df.b
df["ev_n"] = df.groupby("event")["t"].transform("count"); df["mb"] = np.minimum((df.mid // 10).astype(int), 9)
def boot(sub, draws=2000):
    g = sub.groupby("event")["y"].agg(["sum", "count"]); S, N = g["sum"].to_numpy(float), g["count"].to_numpy(); k = len(S)
    out = np.empty(draws)
    for i in range(0, draws, 400):
        m = min(400, draws - i); idx = rng.integers(0, k, size=(m, k)); out[i:i+m] = S[idx].sum(1) / N[idx].sum(1)
    return S.sum() / N.sum(), *np.percentile(out, [2.5, 97.5])
groups = {"all": df.spread > 0, "spread<=50": df.spread <= 50,
          "spread>=51, 1-2 markets": (df.spread >= 51) & (df.ev_n <= 2),
          "spread>=51, 3+ markets": (df.spread >= 51) & (df.ev_n >= 3)}
R = {"reliability": {}, "mechanism": {}}
for lab, m in groups.items():
    out = []
    for bk in range(10):
        sb = df[m & (df.mb == bk)]
        if len(sb) < 50: continue
        p, lo, hi = boot(sb)
        out.append({"bucket": f"{10*bk}-{10*bk+9}", "n": int(len(sb)), "implied": round(float(sb.mid.mean()), 1),
                    "yes": round(100 * p, 1), "ci": [round(100 * lo, 1), round(100 * hi, 1)]})
    R["reliability"][lab] = out
    sub = df[m]
    R["mechanism"][lab] = {"n": int(len(sub)), "share": round(float(m.mean()), 4), "mean_mid": round(float(sub.mid.mean()), 1),
                           "yes_rate": round(100 * float(sub.y.mean()), 1),
                           "share_mid_40_60": round(float(((sub.mid >= 40) & (sub.mid < 60)).mean()), 3)}
    print(f"{lab:<26} n={len(sub):>6} ({100*m.mean():4.1f}%)  mean mid {sub.mid.mean():5.1f}  YES {100*sub.y.mean():5.1f}%  mid in [40,60): {100*((sub.mid>=40)&(sub.mid<60)).mean():4.1f}%")
json.dump(R, open(f"{OUT}/figure_data.json", "w"), indent=1)
print("saved figure_data.json")
