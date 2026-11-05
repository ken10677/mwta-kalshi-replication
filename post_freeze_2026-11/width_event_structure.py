#!/usr/bin/env python3
"""
Book width, event structure, and midpoint calibration (post-freeze, exploratory).
Swan (2026), "Who Said Fifty?" Every number in Sections 3-4 of the JEF manuscript
that is not in the frozen exhibits comes from this script.

Reads the frozen data (data-freeze-v1): logs/retro_pairs.jsonl for the retrospective
stratum and logs/market_snapshots.jsonl + logs/resolutions.jsonl for the prospective
stratum. Same filters, fee rule, and cluster definitions as tools/gate_analysis.py.

Inference: event-clustered percentile bootstrap, 10,000 draws (numpy PCG64, seed
20260824). Within-series comparisons use 1,000 draws.

Usage: python width_event_structure.py [LOGS_DIR] [OUT_DIR]
"""
import json, math, re, sys
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd

LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
DRAWS, FE_DRAWS = 10000, 1000
rng = np.random.default_rng(20260824)


def fee(cost, mult):
    q = cost / 100.0
    return math.ceil(mult * q * (1 - q) * 100)


# ---------------------------------------------------------------- retrospective
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"):
        continue
    b, a = int(r.get("b") or 0), int(r.get("a") or 100)
    p = r.get("p") or round((b + a) / 2)
    tth = None
    try:
        cd = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
        if r.get("qts"):
            tth = (cd.timestamp() - float(r["qts"])) / 3600
    except Exception:
        pass
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "",
                 "b": b, "a": a, "p": int(p), "y": 1 if r["r"] == "yes" else 0, "tth": tth})
df = pd.DataFrame(rows)
df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].reset_index(drop=True)
assert len(df) == 57808 and df.event.nunique() == 19718
df["mid"] = (df.a + df.b) / 2
df["spread"] = df.a - df.b
df["gap"] = df.y * 100 - df.mid
df["mb"] = np.minimum((df.mid // 10).astype(int), 9)
df["db"] = np.minimum(df.p // 10, 9)
df["ev_n"] = df.groupby("event")["t"].transform("count")
for m in (0.0, 0.07):
    df[f"no_{m}"] = [(100 if y == 0 else 0) - (100 - b) - fee(100 - b, m) for y, b in zip(df.y, df.b)]
    df[f"yes_{m}"] = [(100 if y == 1 else 0) - a - fee(a, m) for y, a in zip(df.y, df.a)]


def boot(sub, col="gap", draws=DRAWS):
    g = sub.groupby("event")[col].agg(["sum", "count"])
    S, N = g["sum"].to_numpy(float), g["count"].to_numpy()
    k = len(S)
    if k == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0, "clusters": 0}
    out = np.empty(draws)
    for i in range(0, draws, 250):
        m = min(250, draws - i)
        idx = rng.integers(0, k, size=(m, k))
        out[i:i + m] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"est": round(float(S.sum() / N.sum()), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "n": int(len(sub)), "clusters": int(k)}


def slope_int_brier(sub):
    x = sub.mid.to_numpy() / 100
    y = sub.y.to_numpy().astype(float)
    s = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return {"slope": round(float(s), 3), "intercept": round(float(y.mean() - s * x.mean()), 3),
            "brier": round(float(((x - y) ** 2).mean()), 4)}


def fe_wide_minus_narrow(d, draws=FE_DRAWS):
    """Within series x midpoint-bucket cells: OLS of gap on 1[spread >= 51] with cell fixed effects."""
    d = d.reset_index(drop=True)
    cell = pd.factorize(d.series + "|" + d.mb.astype(str))[0]
    w = (d.spread >= 51).to_numpy(float)
    g = d.gap.to_numpy(float)
    ev_codes, ev_idx = np.unique(pd.factorize(d.event)[0], return_inverse=True)
    order = np.argsort(ev_idx, kind="stable")
    bounds = np.searchsorted(ev_idx[order], np.arange(len(ev_codes) + 1))

    def est(ix):
        c, ww, gg = cell[ix], w[ix], g[ix]
        nc = np.bincount(c, minlength=cell.max() + 1)
        wm = np.bincount(c, ww, cell.max() + 1)[c] / nc[c]
        gm = np.bincount(c, gg, cell.max() + 1)[c] / nc[c]
        dw = ww - wm
        den = (dw * dw).sum()
        return float((dw * (gg - gm)).sum() / den) if den > 0 else np.nan

    pt = est(np.arange(len(d)))
    informative = int(pd.Series(w).groupby(cell).agg(lambda s: 0 < s.mean() < 1).sum())
    bs = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(ev_codes), len(ev_codes))
        ix = np.concatenate([order[bounds[e]:bounds[e + 1]] for e in pick])
        bs[i] = est(ix)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": round(pt, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "draws": draws, "informative_cells": informative, "n": int(len(d))}


R, L = {}, []


def P(s=""):
    print(s, flush=True)
    L.append(s)


fmt = lambda b: f"{b['est']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] n={b['n']} clusters={b['clusters']}"

NAR, WID = df.spread <= 50, df.spread >= 51
SMALL, LARGE = df.ev_n <= 2, df.ev_n >= 3
SEG = {"all": df.spread > 0, "spread<=50": NAR, "spread>=51": WID,
       "spread>=51 & 1-2 mkts": WID & SMALL, "spread>=51 & 3+ mkts": WID & LARGE,
       "all except (spread>=51 & 3+)": ~(WID & LARGE),
       "spread<=50 & 1-2 mkts": NAR & SMALL, "spread<=50 & 3+ mkts": NAR & LARGE}

P("[1] Composition")
R["composition"] = {}
for k, m in SEG.items():
    s = df[m]
    R["composition"][k] = {"n": int(m.sum()), "share": round(float(m.mean()), 4), "clusters": int(s.event.nunique()),
                           "mean_mid": round(float(s.mid.mean()), 2), "yes_rate": round(float(100 * s.y.mean()), 2),
                           "mean_bid": round(float(s.b.mean()), 2), "mean_ask": round(float(s.a.mean()), 2),
                           "share_mid_40_60": round(float(((s.mid >= 40) & (s.mid < 60)).mean()), 3)}
    c = R["composition"][k]
    P(f"  {k:<30} n={c['n']:>6} ({100*c['share']:.1f}%) clusters={c['clusters']:>5} mean mid {c['mean_mid']:.1f} "
      f"YES {c['yes_rate']:.1f}% bid {c['mean_bid']:.1f} ask {c['mean_ask']:.1f} mids in [40,60) {100*c['share_mid_40_60']:.1f}%")
nb = (df.b <= 5) & (df.a >= 90)
R["near_empty_bid5_ask90"] = {"n": int(nb.sum()), "share": round(float(nb.mean()), 4)}
P(f"  near-empty (bid<=5 & ask>=90): {int(nb.sum())} ({100*nb.mean():.1f}%)")
top = Counter(df[WID & LARGE].series).most_common(8)
R["top_series_wide_3plus"] = [{"series": s, "n": n, "share": round(n / int((WID & LARGE).sum()), 3)} for s, n in top]
P("  top series in spread>=51 & 3+: " + ", ".join(f"{s} {n}" for s, n in top))
soccer = df.series.str.startswith("KXCLUBF")
R["club_soccer_share_of_wide_3plus"] = round(float((soccer & WID & LARGE).sum() / (WID & LARGE).sum()), 3)
P(f"  club-soccer share of spread>=51 & 3+: {100*R['club_soccer_share_of_wide_3plus']:.1f}%")

P("\n[2] Midpoint calibration gap by segment")
R["gap"] = {}
for k, m in SEG.items():
    b = boot(df[m]); b.update(slope_int_brier(df[m])); R["gap"][k] = b
    P(f"  {k:<30} {fmt(b)} slope {b['slope']:.3f} int {b['intercept']:+.3f} Brier {b['brier']:.4f}")
seg_contrib = R["composition"]["spread>=51 & 3+ mkts"]["share"] * R["gap"]["spread>=51 & 3+ mkts"]["est"]
R["contribution_wide3_to_aggregate"] = round(seg_contrib, 2)
P(f"  share x gap of spread>=51 & 3+ segment: {seg_contrib:+.2f} pp of the aggregate {R['gap']['all']['est']:+.2f}")
R["midcurve"] = {}
for k in ("all", "spread<=50", "spread>=51 & 1-2 mkts", "spread>=51 & 3+ mkts", "all except (spread>=51 & 3+)"):
    b = boot(df[SEG[k] & (df.mid >= 30) & (df.mid < 60)]); R["midcurve"][k] = b
    P(f"  mid-curve [30,60) {k:<28} {fmt(b)}")

P("\n[3] Gap by spread bin x event size (Table 2)")
R["bins"] = []
for lo, hi in ((1, 4), (5, 20), (21, 50), (51, 70), (71, 99)):
    m = (df.spread >= lo) & (df.spread <= hi)
    row = {"bin": f"{lo}-{hi}", "share": round(float(m.mean()), 3), "all": boot(df[m]),
           "1-2": boot(df[m & SMALL]), "3+": boot(df[m & LARGE])}
    R["bins"].append(row)
    P(f"  {row['bin']:>6} share {100*row['share']:.1f}% | all {fmt(row['all'])} | 1-2 {fmt(row['1-2'])} | 3+ {fmt(row['3+'])}")
row = {"bin": "all", "share": 1.0, "all": R["gap"]["all"], "1-2": boot(df[SMALL]), "3+": boot(df[LARGE])}
R["bins"].append(row)
P(f"     all | 1-2 {fmt(row['1-2'])} | 3+ {fmt(row['3+'])}")

P("\n[4] Robustness")
R["cutoffs"] = []
for c in (21, 31, 41, 51, 61, 71):
    x = {"cutoff": c, "share_narrow": round(float((df.spread < c).mean()), 3),
         "narrow": boot(df[df.spread < c]), "wide": boot(df[df.spread >= c])}
    R["cutoffs"].append(x)
    P(f"  cutoff {c}: narrow ({100*x['share_narrow']:.1f}%) {fmt(x['narrow'])} | wide {fmt(x['wide'])}")
R["drop_family"] = []
fams = {"ITF tennis": df.series.isin(["KXITFMATCH", "KXITFWMATCH"]), "CS2 esports": df.series.str.startswith("KXCS2"),
        "club soccer": soccer}
for name, mask in fams.items():
    s = df[~mask]
    x = {"dropped": name, "n_dropped": int(mask.sum()), "narrow": boot(s[s.spread <= 50]),
         "wide_1-2": boot(s[(s.spread >= 51) & (s.ev_n <= 2)]), "wide_3+": boot(s[(s.spread >= 51) & (s.ev_n >= 3)])}
    R["drop_family"].append(x)
    P(f"  drop {name} ({x['n_dropped']}): narrow {fmt(x['narrow'])} | wide 1-2 {fmt(x['wide_1-2'])} | wide 3+ {fmt(x['wide_3+'])}")
R["horizon"] = []
for name, lo, hi in (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 1e12)):
    s = df[df.tth.notna() & (df.tth >= lo) & (df.tth < hi)]
    x = {"stratum": name, "all": boot(s), "narrow": boot(s[s.spread <= 50]), "wide": boot(s[s.spread >= 51])}
    R["horizon"].append(x)
    P(f"  horizon {name}: all {fmt(x['all'])} | narrow {fmt(x['narrow'])} | wide {fmt(x['wide'])}")
P("  within series x midpoint bucket, wide(>=51) minus narrow(<=50):")
R["within_series"] = {}
for k, m in (("all", df.spread > 0), ("1-2 mkts", SMALL), ("3+ mkts", LARGE)):
    x = fe_wide_minus_narrow(df[m]); R["within_series"][k] = x
    P(f"    {k:<9} {x['est']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] informative cells {x['informative_cells']} ({x['draws']} draws)")

P("\n[5] Taker returns by segment (cents/contract, hold to settlement)")
R["taker"] = {}
TSEG = {"all": SEG["all"], "spread<=4": df.spread <= 4, "spread 5-20": (df.spread >= 5) & (df.spread <= 20),
        "spread<=50": NAR, "spread>=51": WID, "spread>=51 & 3+ mkts": WID & LARGE}
for k, m in TSEG.items():
    row = {"n": int(m.sum())}
    for side in ("no", "yes"):
        for mult in (0.0, 0.07):
            row[f"{side}_{mult}"] = boot(df[m], f"{side}_{mult}")
    R["taker"][k] = row
    P(f"  {k:<22} NO fee0 {fmt(row['no_0.0'])}\n  {'':<22} NO .07  {fmt(row['no_0.07'])}\n"
      f"  {'':<22} YES fee0 {fmt(row['yes_0.0'])}\n  {'':<22} YES .07  {fmt(row['yes_0.07'])}")
P("  gate check inside spread<=50 books, display-price buckets as in frozen Table 3 (n>=200, clusters>=30, CI lo>0), fee 0.07:")
R["gate_narrow50"] = []
for side in ("no", "yes"):
    for bk in range(10):
        s = df[NAR & (df.db == bk)]
        b = boot(s, f"{side}_0.07")
        b.update({"side": side, "bucket": f"{10*bk}-{10*bk+9}",
                  "passes": bool(b["n"] >= 200 and b["clusters"] >= 30 and b["lo"] > 0)})
        R["gate_narrow50"].append(b)
        P(f"    BUY {side.upper():<3} display {b['bucket']:<5} {fmt(b)}{'  PASSES' if b['passes'] else ''}")
R["gate_narrow50_any_pass"] = any(x["passes"] for x in R["gate_narrow50"])
R["gate_narrow50_max_hi"] = max(x["hi"] for x in R["gate_narrow50"])
P(f"  any pass: {R['gate_narrow50_any_pass']}; largest upper bound {R['gate_narrow50_max_hi']:+.2f}")

P("\n[6] Reliability by midpoint bucket (Figure 1)")
R["reliability"] = {}
for k in ("all", "spread<=50", "spread>=51 & 1-2 mkts", "spread>=51 & 3+ mkts"):
    out = []
    s = df[SEG[k]].copy(); s["y100"] = s.y * 100
    for bk in range(10):
        c = s[s.mb == bk]
        if len(c) < 30:
            continue
        b = boot(c, "y100", draws=2000)
        out.append({"bucket": bk, "n": int(len(c)), "implied": round(float(c.mid.mean()), 2),
                    "yes": b["est"], "lo": b["lo"], "hi": b["hi"], "clusters": b["clusters"]})
    R["reliability"][k] = out
    P(f"  {k}: " + "; ".join(f"{o['bucket']*10}s {o['implied']:.1f}->{o['yes']:.1f} (n={o['n']})" for o in out))

P("\n[7] Prospective stratum (two-sided books, earliest snapshot) by width")
res = {}
for line in open(f"{LOGS}/resolutions.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") in ("yes", "no"):
        res[r["t"]] = r["r"]
first = {}
for line in open(f"{LOGS}/market_snapshots.jsonl", encoding="utf-8"):
    s = json.loads(line)
    if s["t"] not in first:
        first[s["t"]] = s
TH = re.compile(r"-[TB][\d.\-]+$")
pr = pd.DataFrame([{"t": t, "event": TH.sub("", t), "b": s.get("b") or 0, "a": s.get("a") or 100, "p": s["p"],
                    "y": 1 if res[t] == "yes" else 0} for t, s in first.items() if t in res])
assert len(pr) == 937
pr = pr[(pr.b > 0) & (pr.a < 100) & (pr.a > pr.b)].copy()
pr["mid"] = (pr.a + pr.b) / 2; pr["spread"] = pr.a - pr.b; pr["gap"] = pr.y * 100 - pr.mid
pr["ev_n"] = pr.groupby("event")["t"].transform("count")
R["prospective"] = {"n_two_sided": int(len(pr)), "all": boot(pr), "spread<=50": boot(pr[pr.spread <= 50]),
                    "spread>=51": boot(pr[pr.spread >= 51]),
                    "share_spread>=51": round(float((pr.spread >= 51).mean()), 3),
                    "share_3plus": round(float((pr.ev_n >= 3).mean()), 3)}
for k in ("all", "spread<=50", "spread>=51"):
    P(f"  {k:<11} {fmt(R['prospective'][k])}")
P(f"  share spread>=51: {R['prospective']['share_spread>=51']}; share in 3+ market clusters: {R['prospective']['share_3plus']}")

json.dump(R, open(f"{OUT}/width_event_structure.json", "w"), indent=1)
open(f"{OUT}/width_event_structure.txt", "w").write("\n".join(L) + "\n")
P(f"\nartifacts: {OUT}/width_event_structure.json, .txt")
