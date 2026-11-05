#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), added after the pre-registered political results.
Swan (2026), "Who Said Fifty Percent?". Same entry rule, filters, and inference as the
Study 1 and Study 2 scripts: event-clustered percentile bootstrap, 2,000 draws, numpy PCG64,
seed 20260824 (a fresh generator per study).

  A. Slope, intercept, crossing point, and mean gap by spread x event size, both studies.
  B. Mechanism check: wide-book calibration at the bid, at the ask, and at the midpoint.
  C. Political decomposition of the mean gap; narrow books with midpoints of 30-59 cents.
  D. Political mean gap by time from entry to close.
  E. Entry-spread distributions (for Supplementary Figure S1).

Usage: python EXPLORATORY_jdm_checks.py STUDY1_RETRO_PAIRS_JSONL STUDY2_ENTRIES_JSONL OUT_JSON"""
import json, sys
import numpy as np, pandas as pd

S1, S2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DRAWS = 2000


def load_s1(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") not in ("yes", "no"):
            continue
        b, a = int(r.get("b") or 0), int(r.get("a") or 100)
        rows.append({"t": r["t"], "event": r.get("event") or r["t"], "b": b, "a": a, "y": 1.0 if r["r"] == "yes" else 0.0})
    d = pd.DataFrame(rows)
    d = d[(d.b > 0) & (d.a < 100) & (d.a > d.b)].reset_index(drop=True)
    assert len(d) == 57808 and d.event.nunique() == 19718
    d["ev_n"] = d.groupby("event")["t"].transform("count")
    return d


def load_s2(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d = d[d.status == "ok"].copy().reset_index(drop=True)
    d["y"] = (d.result == "yes").astype(float)
    d["ev_n"] = d.groupby("event")["ticker"].transform("count")
    d["days"] = (d.close - d.entry_ts) / 86400
    assert len(d) == 6922
    return d


def prep(d):
    d["mid"] = (d.a + d.b) / 2
    d["spread"] = d.a - d.b
    return d


def stats(sub, col, rng):
    x, y = sub[col].to_numpy(float) / 100, sub.y.to_numpy(float)
    g = pd.DataFrame({"e": sub.event.to_numpy(), "n": 1.0, "x": x, "y": y, "xx": x * x, "xy": x * y,
                      "gap": y * 100 - x * 100}).groupby("e").sum().to_numpy()

    def est(t):
        n, sx, sy, sxx, sxy, sg = (t[..., i] for i in range(6))
        slope = (sxy - sx * sy / n) / (sxx - sx * sx / n)
        return slope, sy / n - slope * sx / n, sg / n

    s0, i0, g0 = est(g.sum(0))
    k = len(g)
    bs = np.empty((DRAWS, 3))
    for j in range(0, DRAWS, 200):
        m = min(200, DRAWS - j)
        idx = rng.integers(0, k, size=(m, k))
        bs[j:j + m] = np.column_stack(est(g[idx].sum(1)))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    cross = (-i0 / (s0 - 1)) * 100 if abs(s0 - 1) > 1e-9 else None
    return {"n": int(len(sub)), "events": int(k), "slope": round(float(s0), 3),
            "slope_ci": [round(float(lo[0]), 3), round(float(hi[0]), 3)], "intercept": round(float(i0), 3),
            "crosses_diagonal_at": None if cross is None else round(float(cross), 1),
            "gap": round(float(g0), 2), "gap_ci": [round(float(lo[2]), 2), round(float(hi[2]), 2)],
            "mean_price": round(float(sub[col].mean()), 2), "yes_rate": round(float(sub.y.mean() * 100), 2)}


def slope_diff(sub, c1, c2, rng):
    """Paired event-clustered bootstrap of slope(c1) - slope(c2) on the same resampled events."""
    y = sub.y.to_numpy(float)
    parts = []
    for c in (c1, c2):
        x = sub[c].to_numpy(float) / 100
        parts.append(pd.DataFrame({"e": sub.event.to_numpy(), "n": 1.0, "x": x, "y": y, "xx": x * x, "xy": x * y}).groupby("e").sum())
    G = [p.to_numpy() for p in parts]

    def sl(t):
        n, sx, sy, sxx, sxy = (t[..., i] for i in range(5))
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)

    d0 = sl(G[0].sum(0)) - sl(G[1].sum(0))
    k = len(G[0])
    bs = np.empty(DRAWS)
    for j in range(0, DRAWS, 200):
        m = min(200, DRAWS - j)
        idx = rng.integers(0, k, size=(m, k))
        bs[j:j + m] = sl(G[0][idx].sum(1)) - sl(G[1][idx].sum(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": round(float(d0), 3), "ci": [round(float(lo), 3), round(float(hi), 3)]}


def cells(d):
    nar, wid = d[d.spread <= 50], d[d.spread >= 51]
    return {"all": d, "narrow": nar, "narrow_events1-2": nar[nar.ev_n <= 2], "narrow_events3+": nar[nar.ev_n >= 3],
            "wide": wid, "wide_events1-2": wid[wid.ev_n <= 2], "wide_events3+": wid[wid.ev_n >= 3]}


R = {}
for name, loader, path in (("study1", load_s1, S1), ("study2", load_s2, S2)):
    rng = np.random.default_rng(20260824)
    d = prep(loader(path))
    C = cells(d)
    out = {"A_cells_midpoint": {k: stats(v, "mid", rng) for k, v in C.items()}}
    w = C["wide"]
    out["B_wide_mechanism"] = {
        "at_bid": stats(w, "b", rng), "at_ask": stats(w, "a", rng), "at_mid": stats(w, "mid", rng),
        "sd_bid": round(float(w.b.std()), 2), "sd_ask": round(float(w.a.std()), 2),
        "share_bid_le5": round(float((w.b <= 5).mean()), 3), "share_ask_ge95": round(float((w.a >= 95).mean()), 3),
        "share_bid_le5_or_ask_ge95": round(float(((w.b <= 5) | (w.a >= 95)).mean()), 3)}
    for k in ("wide_events1-2", "wide_events3+"):
        out["B_wide_mechanism"][k] = {"at_bid": stats(C[k], "b", rng), "at_ask": stats(C[k], "a", rng)}
    out["B_wide_mechanism"]["slope_mid_minus_bid"] = slope_diff(w, "mid", "b", rng)
    out["B_wide_mechanism"]["slope_mid_minus_ask"] = slope_diff(w, "mid", "a", rng)
    out["B_wide_mechanism"]["corr_bid_ask"] = round(float(np.corrcoef(w.b, w.a)[0, 1]), 3)
    out["E_spread_counts"] = {int(s): int(c) for s, c in d.spread.round().astype(int).value_counts().sort_index().items()}  # sub-cent spreads rounded to the nearest cent
    if name == "study2":
        d["gap"] = d.y * 100 - d.mid
        tot = d.gap.sum()
        out["C_decomposition"] = {k: {"n": int(len(v)), "share_of_markets": round(len(v) / len(d), 3),
                                       "share_of_total_gap": round(float((v.y * 100 - v.mid).sum() / tot), 3)}
                                  for k, v in C.items() if k != "all"}
        nb = C["narrow"]
        out["C_narrow_mid30_59"] = stats(nb[(nb.mid >= 30) & (nb.mid < 60)], "mid", rng)
        bins = [(-1e9, 1, "<1 day"), (1, 7, "1-7 days"), (7, 30, "7-30 days"), (30, 1e9, ">=30 days")]
        out["D_horizon"] = {"median_days": round(float(d.days.median()), 1)}
        for lo_, hi_, lab in bins:
            h = d[(d.days >= lo_) & (d.days < hi_)]
            out["D_horizon"][lab] = {"all": stats(h, "mid", rng), "narrow": stats(h[h.spread <= 50], "mid", rng),
                                     "wide_n": int((h.spread >= 51).sum())}
    R[name] = out

json.dump(R, open(OUT, "w"), indent=1)
for s in R:
    print("==", s)
    for k, v in R[s]["A_cells_midpoint"].items():
        print(f"  {k:<18} n={v['n']:>6} slope {v['slope']} {v['slope_ci']} int {v['intercept']} cross {v['crosses_diagonal_at']} gap {v['gap']} {v['gap_ci']}")
    B = R[s]["B_wide_mechanism"]
    for k in ("at_bid", "at_ask", "at_mid"):
        v = B[k]; print(f"  wide {k:<7} slope {v['slope']} {v['slope_ci']} int {v['intercept']} gap {v['gap']} mean {v['mean_price']}")
    print("  mid-bid", B["slope_mid_minus_bid"], "mid-ask", B["slope_mid_minus_ask"], "corr", B["corr_bid_ask"])
    print("  sd bid/ask", B["sd_bid"], B["sd_ask"], "bid<=5", B["share_bid_le5"], "ask>=95", B["share_ask_ge95"], "either", B["share_bid_le5_or_ask_ge95"])
    for k in ("wide_events1-2", "wide_events3+"):
        print("  ", k, "bid", B[k]["at_bid"]["slope"], B[k]["at_bid"]["slope_ci"], "ask", B[k]["at_ask"]["slope"], B[k]["at_ask"]["slope_ci"])
print(json.dumps(R["study2"]["C_decomposition"], indent=0))
print("narrow 30-59", R["study2"]["C_narrow_mid30_59"])
print(json.dumps(R["study2"]["D_horizon"], indent=0)[:3000])
