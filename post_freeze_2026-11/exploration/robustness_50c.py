#!/usr/bin/env python3
"""Robustness of the 50-cent book-width split (post-freeze, exploratory). Swan (2026).
Usage: python robustness_50c.py [LOGS_DIR] [OUT_DIR]"""
import json, sys
from datetime import datetime
import numpy as np, pandas as pd
LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rng = np.random.default_rng(20260824)
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"): continue
    b, a = int(r.get("b") or 0), int(r.get("a") or 100); tth = None
    try:
        cd = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
        if r.get("qts"): tth = (cd.timestamp() - float(r["qts"])) / 3600
    except Exception: pass
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "", "b": b, "a": a,
                 "y": 1 if r["r"] == "yes" else 0, "tth": tth})
df = pd.DataFrame(rows); df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy()
df["mid"] = (df.a + df.b) / 2; df["spread"] = df.a - df.b; df["gap"] = df.y * 100 - df.mid
df["mb"] = np.minimum((df.mid // 10).astype(int), 9); df["ev_n"] = df.groupby("event")["t"].transform("count")
def boot(sub, draws=2000):
    g = sub.groupby("event")["gap"].agg(["sum", "count"]); S, N = g["sum"].to_numpy(float), g["count"].to_numpy(); k = len(S)
    if k == 0: return [np.nan] * 3 + [0, 0]
    out = np.empty(draws)
    for i in range(0, draws, 400):
        m = min(400, draws - i); idx = rng.integers(0, k, size=(m, k)); out[i:i+m] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5]); return [round(float(S.sum() / N.sum()), 2), round(float(lo), 2), round(float(hi), 2), int(len(sub)), int(k)]
def fe(d, draws=300):
    d = d.copy(); d["w"] = (d.spread >= 51).astype(float); d["cell"] = d.series + "|" + d.mb.astype(str)
    def est(x):
        gm = x.groupby("cell").gap.transform("mean"); wm = x.groupby("cell").w.transform("mean"); dw = x.w - wm
        den = (dw * dw).sum(); return float((dw * (x.gap - gm)).sum() / den) if den > 0 else np.nan
    pt = est(d); ev = d.event.unique(); idx = d.groupby("event").indices; bs = []
    for _ in range(draws):
        pick = rng.choice(ev, len(ev)); ix = np.concatenate([idx[e] for e in pick]); bs.append(est(d.iloc[ix]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return [round(pt, 2), round(float(lo), 2), round(float(hi), 2), draws]
R = {}; nar, wid = df.spread <= 50, df.spread >= 51
R["cutoffs"] = []
for c in (21, 31, 41, 51, 61, 71):
    n_, w_ = df[df.spread < c], df[df.spread >= c]
    R["cutoffs"].append({"c": c, "narrow": boot(n_), "wide": boot(w_), "share_narrow": round(float((df.spread < c).mean()), 3)})
R["drop_series"] = []
fam = {"ITF tennis": df.series.isin(["KXITFMATCH", "KXITFWMATCH"]), "CS2 esports": df.series.str.startswith("KXCS2"),
       "club soccer": df.series.str.startswith("KXCLUBF")}
for name, mask in fam.items():
    s = df[~mask]; R["drop_series"].append({"dropped": name, "narrow": boot(s[s.spread <= 50]), "wide": boot(s[s.spread >= 51])})
R["horizon"] = []
for name, lo, hi in (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 1e9)):
    s = df[df.tth.notna() & (df.tth >= lo) & (df.tth < hi)]
    R["horizon"].append({"stratum": name, "all": boot(s), "narrow": boot(s[s.spread <= 50]), "wide": boot(s[s.spread >= 51])})
R["fe_within_series_bucket"] = {"all": fe(df), "1-2 markets": fe(df[df.ev_n <= 2]), "3+ markets": fe(df[df.ev_n >= 3])}
R["bins_by_eventsize"] = []
for lo, hi in ((1, 4), (5, 20), (21, 50), (51, 70), (71, 99)):
    s = df[(df.spread >= lo) & (df.spread <= hi)]
    R["bins_by_eventsize"].append({"bin": f"{lo}-{hi}", "all": boot(s), "1-2": boot(s[s.ev_n <= 2]), "3+": boot(s[s.ev_n >= 3]),
                                   "share": round(float(len(s) / len(df)), 3)})
json.dump(R, open(f"{OUT}/robustness_50c.json", "w"), indent=1)
for k, v in R.items():
    print(k); 
    if isinstance(v, list): [print("  ", x) for x in v]
    else: [print("  ", kk, vv) for kk, vv in v.items()]
