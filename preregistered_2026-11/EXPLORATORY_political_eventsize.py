#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): political markets by spread x event size, and share of wide-book
midpoints between 40 and 60. Swan (2026). Usage: python EXPLORATORY_political_eventsize.py ENTRIES_JSONL OUT_JSON"""
import json, sys
import numpy as np, pandas as pd
rng = np.random.default_rng(20260824)
d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1])]); d = d[d.status == "ok"].copy()
d["y"] = (d.result == "yes") * 100.0; d["mid"] = (d.a + d.b) / 2; d["spread"] = d.a - d.b; d["gap"] = d.y - d.mid
d["ev_n"] = d.groupby("event")["ticker"].transform("count")
def boot(s):
    g = s.groupby("event").gap.agg(["sum", "count"]); S, N = g["sum"].to_numpy(), g["count"].to_numpy(); k = len(S)
    idx = rng.integers(0, k, size=(2000, k)); b = S[idx].sum(1) / N[idx].sum(1)
    return {"gap": round(float(S.sum() / N.sum()), 2), "ci": [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)],
            "n": int(len(s)), "events": int(k), "mean_mid": round(float(s.mid.mean()), 1), "yes": round(float(s.y.mean()), 1)}
w, n = d[d.spread >= 51], d[d.spread <= 50]
R = {"wide_events1-2": boot(w[w.ev_n <= 2]), "wide_events3+": boot(w[w.ev_n >= 3]),
     "narrow_events1-2": boot(n[n.ev_n <= 2]), "narrow_events3+": boot(n[n.ev_n >= 3]),
     "share_wide_mid_40_60": round(float(((w.mid >= 40) & (w.mid < 60)).mean()), 3)}
json.dump(R, open(sys.argv[2], "w"), indent=1); print(json.dumps(R))
