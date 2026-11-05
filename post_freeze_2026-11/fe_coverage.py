#!/usr/bin/env python3
"""How much of the wide multi-market segment can a within-series comparison see? (post-freeze). Swan (2026).
Usage: python fe_coverage.py [LOGS_DIR] [OUT_DIR]"""
import json, sys
import numpy as np, pandas as pd
LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"): continue
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "",
                 "b": int(r.get("b") or 0), "a": int(r.get("a") or 100), "y": 1 if r["r"] == "yes" else 0})
df = pd.DataFrame(rows); df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy(); assert len(df) == 57808
df["mid"] = (df.a + df.b) / 2; df["spread"] = df.a - df.b; df["gap"] = df.y * 100 - df.mid
df["mb"] = np.minimum((df.mid // 10).astype(int), 9); df["ev_n"] = df.groupby("event")["t"].transform("count")
df["cell"] = df.series + "|" + df.mb.astype(str); df["wide"] = df.spread >= 51
mixed = df.groupby("cell").wide.agg(lambda s: 0 < s.mean() < 1)
df["mixed"] = df.cell.map(mixed)
seg = df[df.wide & (df.ev_n >= 3)]
R = {"segment_n": int(len(seg)), "segment_share_in_mixed_cells": round(float(seg.mixed.mean()), 3)}
m = df[df.mixed & (df.ev_n >= 3)]
R["mixed_cells_3plus"] = {"wide_gap": round(float(m[m.wide].gap.mean()), 2), "narrow_gap": round(float(m[~m.wide].gap.mean()), 2),
                         "wide_n": int(m.wide.sum()), "narrow_n": int((~m.wide).sum())}
s = df[df.series.str.startswith("KXCLUBF")]
R["club_soccer"] = {"n": int(len(s)), "share_wide": round(float(s.wide.mean()), 3),
                    "gap_narrow": round(float(s[~s.wide].gap.mean()), 2), "gap_wide": round(float(s[s.wide].gap.mean()), 2),
                    "midcurve_gap_narrow": round(float(s[~s.wide & (s.mid >= 30) & (s.mid < 60)].gap.mean()), 2),
                    "midcurve_gap_wide": round(float(s[s.wide & (s.mid >= 30) & (s.mid < 60)].gap.mean()), 2),
                    "midcurve_n_narrow": int((~s.wide & (s.mid >= 30) & (s.mid < 60)).sum())}
print(json.dumps(R, indent=1)); json.dump(R, open(f"{OUT}/fe_coverage.json", "w"), indent=1)

# Mid-curve [30,60) gap by width and event size, event-clustered bootstrap (10,000 draws)
rng = np.random.default_rng(20260824)
def boot(sub, draws=10000):
    g = sub.groupby("event")["gap"].agg(["sum", "count"]); S, N = g["sum"].to_numpy(float), g["count"].to_numpy(); k = len(S)
    out = np.empty(draws)
    for i in range(0, draws, 500):
        mm = min(500, draws - i); idx = rng.integers(0, k, size=(mm, k)); out[i:i+mm] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"est": round(float(S.sum()/N.sum()), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(len(sub)), "clusters": int(k)}
mc = df[(df.mid >= 30) & (df.mid < 60)]
R["midcurve_by_width_eventsize"] = {
    "narrow_1-2": boot(mc[(mc.spread <= 50) & (mc.ev_n <= 2)]), "narrow_3+": boot(mc[(mc.spread <= 50) & (mc.ev_n >= 3)]),
    "wide_1-2": boot(mc[(mc.spread >= 51) & (mc.ev_n <= 2)]), "wide_3+": boot(mc[(mc.spread >= 51) & (mc.ev_n >= 3)])}
for k, v in R["midcurve_by_width_eventsize"].items(): print("mid-curve", k, v)
json.dump(R, open(f"{OUT}/fe_coverage.json", "w"), indent=1)
