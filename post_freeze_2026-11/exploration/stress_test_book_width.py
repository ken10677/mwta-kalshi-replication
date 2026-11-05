#!/usr/bin/env python3
"""
Stress tests for the book-width (bid-ask spread) result, post-freeze and exploratory.
Swan (2026). Reads the frozen retrospective stratum
(logs/retro_pairs.jsonl) with the same loader rules as tools/gate_analysis.py.

Usage: python stress_test_book_width.py [LOGS_DIR] [OUT_DIR]
CIs: event-clustered percentile bootstrap, numpy PCG64 seed 20260824.
"""
import json, sys, math
from datetime import datetime
import numpy as np
import pandas as pd

LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 20260824
DRAWS = 2000
rng = np.random.default_rng(SEED)

# ---------- load (port of gate_analysis.load_pairs_retro) ----------
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"):
        continue
    b, a = int(r.get("b") or 0), int(r.get("a") or 100)
    tth = None
    try:
        cd = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
        if r.get("qts"):
            tth = (cd.timestamp() - float(r["qts"])) / 3600
    except Exception:
        pass
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "",
                 "b": b, "a": a, "y": 1 if r["r"] == "yes" else 0, "tth": tth})
df = pd.DataFrame(rows)
df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy()
df["mid"] = (df.a + df.b) / 2
df["spread"] = df.a - df.b
df["gap"] = df.y * 100 - df.mid
df["mbucket"] = np.minimum((df.mid // 10).astype(int), 9)
df["midcurve"] = (df.mid >= 30) & (df.mid < 60)
df["ev_n"] = df.groupby("event")["t"].transform("count")
assert len(df) == 57808, len(df)

def boot_mean(sub, col="gap", draws=DRAWS):
    g = sub.groupby("event")[col].agg(["sum", "count"])
    S, N = g["sum"].to_numpy(), g["count"].to_numpy()
    k = len(S)
    if k == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    pt = S.sum() / N.sum()
    out = np.empty(draws); done = 0
    while done < draws:
        m = min(400, draws - done)
        idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = S[idx].sum(1) / N[idx].sum(1)
        done += m
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (pt, lo, hi, len(sub), k)

def fmt(t):
    return f"{t[0]:+6.2f} [{t[1]:+6.2f}, {t[2]:+6.2f}]  n={t[3]:>6} clus={t[4]:>5}"

R = {}
L = []
def P(s=""):
    print(s); L.append(s)

P("STRESS TESTS: book width and midpoint calibration (retrospective stratum, n=57,808)")
P(f"bootstrap: {DRAWS} event-clustered draws, seed {SEED}\n")

# ---------- A. cutoff sensitivity ----------
P("[A] Cutoff sensitivity: narrow = spread < c, wide = spread >= c")
R["cutoffs"] = []
for c in (6, 11, 16, 21, 31, 41, 61):
    nar, wid = df[df.spread < c], df[df.spread >= c]
    rn, rw = boot_mean(nar), boot_mean(wid)
    mn, mw = boot_mean(nar[nar.midcurve]), boot_mean(wid[wid.midcurve])
    P(f"  c={c:>2}c  narrow {fmt(rn)}  |  wide {fmt(rw)}")
    P(f"         mid-curve narrow {fmt(mn)}  |  wide {fmt(mw)}")
    R["cutoffs"].append({"c": c, "share_narrow": round(len(nar) / len(df), 3),
                         "narrow": [round(x, 2) for x in rn[:3]], "wide": [round(x, 2) for x in rw[:3]],
                         "mid_narrow": [round(x, 2) for x in mn[:3]], "mid_wide": [round(x, 2) for x in mw[:3]],
                         "n_narrow": rn[3], "n_wide": rw[3]})

# ---------- B. gap by fine spread bins ----------
P("\n[B] Gap by spread bin")
bins = [(1, 1), (2, 2), (3, 4), (5, 7), (8, 12), (13, 20), (21, 35), (36, 50), (51, 70), (71, 99)]
R["bins"] = []
for lo_, hi_ in bins:
    sub = df[(df.spread >= lo_) & (df.spread <= hi_)]
    r = boot_mean(sub)
    P(f"  {lo_:>2}-{hi_:<2}c  mean mid {sub.mid.mean():5.1f}  gap {fmt(r)}")
    R["bins"].append({"bin": f"{lo_}-{hi_}", "mean_mid": round(sub.mid.mean(), 1),
                      "gap": [round(x, 2) for x in r[:3]], "n": r[3], "clusters": r[4]})

# ---------- C. within series x price-bucket ----------
P("\n[C] Wide-minus-narrow gap difference within series x midpoint-bucket cells (fixed effects)")
df["wide"] = (df.spread >= 21).astype(float)
df["cell"] = df.series + "|" + df.mbucket.astype(str)
def fe_slope(d):
    gm = d.groupby("cell")["gap"].transform("mean")
    wm = d.groupby("cell")["wide"].transform("mean")
    dg, dw = d.gap - gm, d.wide - wm
    den = (dw * dw).sum()
    return float((dw * dg).sum() / den) if den > 0 else np.nan
pt = fe_slope(df)
ev = df.event.unique(); grp = {e: i for i, e in enumerate(ev)}
df["eid"] = df.event.map(grp)
idx_by_ev = df.groupby("eid").indices
evs = np.array(list(idx_by_ev.keys()))
FE_DRAWS = 500
bs = []
gap_arr, wide_arr, cell_arr = df.gap.to_numpy(), df.wide.to_numpy(), df.cell.to_numpy()
for _ in range(FE_DRAWS):
    pick = rng.choice(evs, size=len(evs), replace=True)
    ix = np.concatenate([idx_by_ev[e] for e in pick])
    d = pd.DataFrame({"gap": gap_arr[ix], "wide": wide_arr[ix], "cell": cell_arr[ix]})
    bs.append(fe_slope(d))
lo, hi = np.nanpercentile(bs, [2.5, 97.5])
cells_both = df.groupby("cell")["wide"].agg(["min", "max"])
n_info = int(((cells_both["min"] == 0) & (cells_both["max"] == 1)).sum())
P(f"  wide-minus-narrow (pp): {pt:+.2f} [{lo:+.2f}, {hi:+.2f}]  ({FE_DRAWS} draws; {n_info} series x bucket cells contain both)")
R["within_series_bucket"] = {"diff": round(pt, 2), "ci": [round(lo, 2), round(hi, 2)], "draws": FE_DRAWS,
                             "informative_cells": n_info}

# ---------- D. drop dominant series ----------
P("\n[D] Dropping dominant series families")
fam = {"tennis (ITF/ITFW)": df.series.isin(["KXITFMATCH", "KXITFWMATCH"]),
       "CS2 esports": df.series.str.startswith("KXCS2"),
       "both": df.series.isin(["KXITFMATCH", "KXITFWMATCH"]) | df.series.str.startswith("KXCS2")}
R["drop_series"] = []
for name, mask in fam.items():
    sub = df[~mask]
    rn, rw = boot_mean(sub[sub.spread <= 20]), boot_mean(sub[sub.spread >= 21])
    P(f"  without {name:<18} narrow {fmt(rn)}  |  wide {fmt(rw)}")
    R["drop_series"].append({"dropped": name, "narrow": [round(x, 2) for x in rn[:3]],
                             "wide": [round(x, 2) for x in rw[:3]], "n": len(sub)})

# ---------- E. horizon control ----------
P("\n[E] Within entry-horizon strata")
R["horizon"] = []
for name, lo_h, hi_h in (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 1e9)):
    sub = df[df.tth.notna() & (df.tth >= lo_h) & (df.tth < hi_h)]
    rn, rw = boot_mean(sub[sub.spread <= 20]), boot_mean(sub[sub.spread >= 21])
    P(f"  {name:>6}  narrow {fmt(rn)}  |  wide {fmt(rw)}")
    R["horizon"].append({"stratum": name, "narrow": [round(x, 2) for x in rn[:3]],
                         "wide": [round(x, 2) for x in rw[:3]], "n_narrow": rn[3], "n_wide": rw[3],
                         "clus_narrow": rn[4], "clus_wide": rw[4]})

# ---------- F. mechanism: event size and near-empty books ----------
P("\n[F] Mechanism checks")
empty = (df.b <= 5) & (df.a >= 90)
P(f"  near-empty books (bid<=5c, ask>=90c): {int(empty.sum())} rows "
  f"({100*empty.mean():.1f}% of all; {100*empty[df.spread>=21].mean():.1f}% of wide books); "
  f"mean midpoint {df[empty].mid.mean():.1f}; YES rate {100*df[empty].y.mean():.1f}%")
re_ = boot_mean(df[empty]); rne = boot_mean(df[(df.spread >= 21) & ~empty])
P(f"  gap, near-empty books:            {fmt(re_)}")
P(f"  gap, wide but not near-empty:     {fmt(rne)}")
R["near_empty"] = {"n": int(empty.sum()), "share_all": round(float(empty.mean()), 3),
                   "share_of_wide": round(float(empty[df.spread >= 21].mean()), 3),
                   "mean_mid": round(float(df[empty].mid.mean()), 1), "yes_rate": round(100 * float(df[empty].y.mean()), 1),
                   "gap": [round(x, 2) for x in re_[:3]], "gap_wide_not_empty": [round(x, 2) for x in rne[:3]],
                   "n_wide_not_empty": rne[3]}
R["event_size"] = []
for name, lo_n, hi_n in (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10**6)):
    sub = df[(df.ev_n >= lo_n) & (df.ev_n <= hi_n)]
    rn, rw = boot_mean(sub[sub.spread <= 20]), boot_mean(sub[sub.spread >= 21])
    P(f"  markets per event {name:>3}: narrow {fmt(rn)}  |  wide {fmt(rw)}")
    R["event_size"].append({"size": name, "narrow": [round(x, 2) for x in rn[:3]],
                            "wide": [round(x, 2) for x in rw[:3]], "n_narrow": rn[3], "n_wide": rw[3]})
# two-outcome match events: sum of midpoints within event
two = df[df.ev_n == 2]
s = two.groupby("event").agg(summid=("mid", "sum"), anywide=("spread", lambda x: (x >= 21).all()),
                             yes=("y", "sum"))
for lab, m in (("both books narrow", ~s.anywide), ("both books wide", s.anywide)):
    ss = s[m]
    P(f"  2-market events, {lab:<17}: events={len(ss):>5}  mean sum of midpoints {ss.summid.mean():6.1f}  "
      f"mean YES per event {ss.yes.mean():.2f}")
R["two_market_events"] = {"narrow": {"events": int((~s.anywide).sum()), "mean_sum_mid": round(float(s[~s.anywide].summid.mean()), 1),
                                     "mean_yes": round(float(s[~s.anywide].yes.mean()), 2)},
                          "wide": {"events": int(s.anywide.sum()), "mean_sum_mid": round(float(s[s.anywide].summid.mean()), 1),
                                   "mean_yes": round(float(s[s.anywide].yes.mean()), 2)}}

# ---------- G. bucket-level reliability data for figure ----------
R["reliability"] = {}
for lab, sub in (("all", df), ("narrow", df[df.spread <= 20]), ("wide", df[df.spread >= 21])):
    out = []
    for bk in range(10):
        sb = sub[sub.mbucket == bk]
        if len(sb) < 30:
            continue
        r = boot_mean(sb, col="y")
        out.append({"bucket": f"{10*bk}-{10*bk+9}", "n": len(sb), "clusters": r[4],
                    "implied": round(float(sb.mid.mean()), 1), "yes": round(100 * r[0], 1),
                    "ci": [round(100 * r[1], 1), round(100 * r[2], 1)]})
    R["reliability"][lab] = out
P("\n[G] Reliability by bucket (narrow <=20c / wide >=21c) saved to JSON")
for lab in ("narrow", "wide"):
    for o in R["reliability"][lab]:
        P(f"  {lab:<6} {o['bucket']:>5}: n={o['n']:>5} implied {o['implied']:5.1f}%  YES {o['yes']:5.1f}% [{o['ci'][0]:5.1f}, {o['ci'][1]:5.1f}]")

json.dump(R, open(f"{OUT}/stress_test_book_width.json", "w"), indent=1)
open(f"{OUT}/stress_test_book_width.txt", "w").write("\n".join(L) + "\n")
print(f"\nartifacts: {OUT}/stress_test_book_width.txt, .json")
