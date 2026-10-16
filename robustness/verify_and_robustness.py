#!/usr/bin/env python3
"""
arXiv-v1 verification + robustness pack for
"Miscalibration Without Tradable Alpha" (Swan, 2026).

POST-FREEZE COMPANION ANALYSIS — does not modify cleanroom-freeze.
Data: cleanroom-freeze/logs/* (data-freeze-v1 artifacts), read-only.

Part 1  VERIFY: recompute, from the raw frozen JSONL, every deterministic
        quantity behind manuscript Tables 1-5 and the in-text prospective
        values; compare against the frozen gate artifacts and the
        manuscript. Loader/bucket/fee/cluster definitions are ported
        line-for-line from tools/gate_analysis.py.
Part 2  NEW (robustness, clearly post-freeze):
        A. Retro spread-stratified midpoint calibration (wide-book confound).
        B. Retro paired display-minus-midpoint gap difference
           (= mean(mid - display); outcomes cancel on identical samples),
           overall and restricted to rows with an explicit candle price.
        C. Prospective display-price calibration restricted to the 583
           two-sided markets + paired wedge (apples-to-apples attenuation).

CIs: event-clustered bootstrap (resample clusters with replacement),
numpy PCG64 seed 20260824, 10,000 draws, percentile 2.5/97.5 — same
procedure as the freeze; RNG engine differs (documented), so frozen CIs
are matched to Monte-Carlo tolerance while all point estimates must
match exactly after rounding.
"""
import json, math, re, sys
from collections import defaultdict
from datetime import datetime
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "logs"
OUTD = sys.argv[2] if len(sys.argv) > 2 else "."
DRAWS = 10_000
SEED = 20260824
rng = np.random.default_rng(SEED)

_THRESH_SUFFIX = re.compile(r"-[TB][\d.\-]+$")
def cluster_key(t): return _THRESH_SUFFIX.sub("", t)

# ---------- loaders: EXACT ports from tools/gate_analysis.py ----------
def load_pairs_prospective():
    res = {}
    for line in open(f"{BASE}/resolutions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") in ("yes", "no"):
            res[r["t"]] = r["r"]
    first = {}
    for line in open(f"{BASE}/market_snapshots.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["t"] not in first:
            first[s["t"]] = s
    pairs = []
    for t, s in first.items():
        if t in res:
            pairs.append({"t": t, "p": s["p"], "b": s.get("b") or 0,
                          "a": s.get("a") or 100, "d": s.get("d"),
                          "yes": 1 if res[t] == "yes" else 0,
                          "cluster": cluster_key(t), "tth": s.get("htc"),
                          "p_explicit": True})
    return pairs

def load_pairs_retro():
    pairs = []
    for line in open(f"{BASE}/retro_pairs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") not in ("yes", "no"):
            continue
        b, a = int(r.get("b") or 0), int(r.get("a") or 100)
        p = r.get("p") or round((b + a) / 2)   # exact port incl. truthiness fallback
        tth = None
        try:
            close_dt = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
            if r.get("qts"):
                tth = (close_dt.timestamp() - float(r["qts"])) / 3600
        except Exception:
            pass
        pairs.append({"t": r["t"], "p": int(p), "b": b, "a": a,
                      "d": (r.get("open") or "")[:10],
                      "yes": 1 if r["r"] == "yes" else 0,
                      "cluster": r.get("event") or cluster_key(r["t"]),
                      "tth": tth, "p_explicit": bool(r.get("p"))})
    return pairs

def fee_cents_mult(cost_cents, mult):
    p = cost_cents / 100.0
    return math.ceil(mult * p * (1 - p) * 100)

def trade_economics(pair, side, mult=0.07):
    b, a = pair["b"], pair["a"]
    if not (0 < b and a < 100 and a > b):
        return None
    if side == "no":
        cost = 100 - b; payout = 100 if pair["yes"] == 0 else 0
    else:
        cost = a; payout = 100 if pair["yes"] == 1 else 0
    return payout - cost - fee_cents_mult(int(cost), mult)

def mid(p): return (p["b"] + p["a"]) / 2
def two_sided_of(pairs): return [p for p in pairs if 0 < p["b"] and p["a"] < 100 and p["a"] > p["b"]]

# ---------- clustered bootstrap (sufficient-stat, vectorized) ----------
def boot_mean_ci(rows, value_of, draws=DRAWS, chunk=500):
    """Clustered bootstrap CI for the pooled mean of value_of(row)."""
    sums, ns = defaultdict(float), defaultdict(int)
    for r in rows:
        v = value_of(r)
        sums[r["cluster"]] += v
        ns[r["cluster"]] += 1
    keys = list(sums.keys())
    S = np.array([sums[k] for k in keys]); N = np.array([ns[k] for k in keys], dtype=np.int64)
    k = len(keys)
    point = S.sum() / N.sum()
    means = np.empty(draws)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        idx = rng.integers(0, k, size=(m, k))
        means[done:done+m] = S[idx].sum(axis=1) / N[idx].sum(axis=1)
        done += m
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, lo, hi, len(rows), k

def boot_slope_ci(rows, x_of, draws=DRAWS, chunk=500):
    """Clustered bootstrap for OLS slope/intercept of yes on x (x in [0,1])."""
    st = defaultdict(lambda: np.zeros(5))  # n, Sx, Sy, Sxy, Sxx
    for r in rows:
        x = x_of(r); y = float(r["yes"])
        s = st[r["cluster"]]
        s += np.array([1.0, x, y, x*y, x*x])
    M = np.array(list(st.values())); k = len(M)
    def sl_ic(T):
        n, sx, sy, sxy, sxx = T[..., 0], T[..., 1], T[..., 2], T[..., 3], T[..., 4]
        denom = sxx - sx * sx / n
        slope = (sxy - sx * sy / n) / denom
        return slope, (sy - slope * sx) / n
    psl, pic = sl_ic(M.sum(axis=0))
    sls = np.empty(draws); ics = np.empty(draws)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        idx = rng.integers(0, k, size=(m, k))
        T = M[idx].sum(axis=1)
        a, b = sl_ic(T)
        sls[done:done+m] = a; ics[done:done+m] = b
        done += m
    return (psl, *np.percentile(sls, [2.5, 97.5])), (pic, *np.percentile(ics, [2.5, 97.5]))

CHECKS = []
def check(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    CHECKS.append((label, got, want, ok))
    return ok

def fmt_ci(t): return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"

def main():
    out = {"generated": datetime.utcnow().isoformat() + "Z",
           "seed": SEED, "draws": DRAWS, "verify": {}, "new": {}}
    R = open(f"{OUTD}/robustness_report.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s); R.write(s + "\n")

    # ================= PART 1: VERIFICATION =================
    P("=" * 78); P("PART 1 — VERIFICATION AGAINST FREEZE + MANUSCRIPT"); P("=" * 78)
    retro = load_pairs_retro()
    check("retro pairs n", len(retro), 57808)
    check("retro clusters", len({p['cluster'] for p in retro}), 19718)
    ts = two_sided_of(retro)
    check("retro all two-sided (display n == midpoint n == 57808)", len(ts), 57808)

    # Table 2 deterministic: n, clusters, implied, actual per mid bucket
    T2_expect = {  # from frozen cleanroom_retro.txt / manuscript Table 2
        "0-9": (3255, 1744, 6.4, 5.4), "10-19": (4345, 2839, 14.8, 13.3),
        "20-29": (5206, 3698, 24.9, 23.7), "30-39": (7920, 5002, 35.5, 30.5),
        "40-49": (15253, 9241, 46.3, 43.5), "50-59": (9480, 6005, 52.9, 50.1),
        "60-69": (4745, 3981, 64.5, 65.0), "70-79": (3768, 2958, 74.4, 75.4),
        "80-89": (2458, 2082, 84.5, 85.7), "90-99": (1378, 1158, 93.1, 94.0)}
    P("\n[Table 2] retro midpoint calibration — deterministic columns:")
    for b0 in range(0, 100, 10):
        bp = [p for p in ts if b0 <= mid(p) <= b0 + 9.99]
        key = f"{b0}-{b0+9}"
        ncl = len({p['cluster'] for p in bp})
        implied = sum(mid(p) for p in bp) / len(bp)
        actual = 100 * sum(p['yes'] for p in bp) / len(bp)
        e = T2_expect[key]
        ok = (check(f"T2 {key} n", len(bp), e[0]) &
              check(f"T2 {key} clusters", ncl, e[1]) &
              check(f"T2 {key} implied", round(implied, 1), e[2]) &
              check(f"T2 {key} actual", round(actual, 1), e[3]))
        P(f"  {key:>6}: n={len(bp):>6} clus={ncl:>5} implied={implied:5.1f}% actual={actual:5.1f}%  {'OK' if ok else '** MISMATCH **'}")

    # Table 1 retro global rows (point estimates deterministic; CIs MC-tolerance)
    P("\n[Table 1] retro global rows:")
    for label, price_of, want_gap, want_sl in (
            ("display", lambda p: float(p["p"]), -2.43, 0.986),
            ("midpoint", mid, -1.98, 1.038)):
        g = boot_mean_ci(ts if label == "midpoint" else retro,
                         lambda r, f=price_of: r["yes"] * 100 - f(r))
        sl, ic = boot_slope_ci(ts if label == "midpoint" else retro,
                               lambda r, f=price_of: f(r) / 100)
        check(f"T1 retro {label} gap", round(g[0], 2), want_gap)
        check(f"T1 retro {label} slope", round(sl[0], 3), want_sl)
        P(f"  {label:>8}: gap {fmt_ci(g[:3])} pp | slope {sl[0]:.3f} [{sl[1]:.3f}, {sl[2]:.3f}] | intercept {ic[0]:.3f}")

    # Table 3 deterministic: mean P/L per bucket-direction (0.07 fee)
    T3_expect = {("no","0-9"):-2.37,("no","10-19"):-2.60,("no","20-29"):-6.38,("no","30-39"):-10.53,
                 ("no","40-49"):-24.42,("no","50-59"):-25.00,("no","60-69"):-7.04,("no","70-79"):-8.19,
                 ("no","80-89"):-10.78,("no","90-99"):-12.72,
                 ("yes","0-9"):-13.60,("yes","10-19"):-8.95,("yes","20-29"):-9.15,("yes","30-39"):-20.70,
                 ("yes","40-49"):-28.63,("yes","50-59"):-30.12,("yes","60-69"):-8.59,("yes","70-79"):-6.58,
                 ("yes","80-89"):-3.76,("yes","90-99"):-5.29}
    P("\n[Table 3] retro taker means (display-bucketed, 0.07 fee):")
    t3_ok = True
    spread_sum = {}
    for side in ("no", "yes"):
        for b0 in range(0, 100, 10):
            bp = [p for p in retro if b0 <= p["p"] <= b0 + 9]
            econ = [(p, trade_economics(p, side)) for p in bp]
            econ = [(p, e) for p, e in econ if e is not None]
            m = sum(e for _, e in econ) / len(econ)
            key = (side, f"{b0}-{b0+9}")
            t3_ok &= check(f"T3 {key}", round(m, 2), T3_expect[key])
            if side == "no":
                spread_sum[f"{b0}-{b0+9}"] = sum(p["a"] - p["b"] for p, _ in econ) / len(econ)
    P(f"  all 20 bucket-direction means: {'OK — match frozen Table 3 exactly' if t3_ok else '** MISMATCH **'}")
    P("  avg entry spread by display bucket (diagnostic): " +
      ", ".join(f"{k}: {v:.1f}c" for k, v in spread_sum.items()))

    # Table 5 horizon deterministic points
    P("\n[Table 5] retro horizon strata:")
    H_expect = {"<24h": (-1.89, 43324, 15467), "24h-7d": (-1.73, 13103, 4139), ">=7d": (-7.15, 1381, 131)}
    for name, lo_h, hi_h in (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 10**9)):
        sub = [p for p in ts if p.get("tth") is not None and lo_h <= p["tth"] < hi_h]
        ncl = len({p['cluster'] for p in sub})
        gm = sum(p["yes"] * 100 - mid(p) for p in sub) / len(sub)
        e = H_expect[name]
        ok = check(f"T5 {name} gap", round(gm, 2), e[0]) & check(f"T5 {name} n", len(sub), e[1]) & check(f"T5 {name} clus", ncl, e[2])
        P(f"  {name:>7}: n={len(sub):>6} clus={ncl:>5} gap={gm:+.2f}pp  {'OK' if ok else '** MISMATCH **'}")

    # Table 4 deterministic points (fee sensitivity, pooled)
    P("\n[Table 4] retro pooled fee sensitivity (point estimates):")
    T4_expect = {(0.0,"no"):-12.99,(0.0,"yes"):-16.95,(0.035,"no"):-13.99,(0.035,"yes"):-17.95,
                 (0.07,"no"):-14.52,(0.07,"yes"):-18.57,(0.14,"no"):-15.54,(0.14,"yes"):-19.72}
    for mult in (0.0, 0.035, 0.07, 0.14):
        line = f"  mult={mult:.3f}:"
        for side in ("no", "yes"):
            vals = [trade_economics(p, side, mult) for p in ts]
            m = sum(v for v in vals if v is not None) / len(ts)
            check(f"T4 {mult} {side}", round(m, 2), T4_expect[(mult, side)])
            line += f"  BUY {side.upper()} {m:+.2f}c"
        P(line)

    # Prospective stratum
    P("\n[Prospective] deterministic verification:")
    pro = load_pairs_prospective()
    check("pro pairs n", len(pro), 937)
    check("pro clusters", len({p['cluster'] for p in pro}), 331)
    pts = two_sided_of(pro)
    check("pro two-sided n", len(pts), 583)
    # the two errata cells, from raw data:
    cell = [p for p in pts if 40 <= mid(p) <= 49.99]
    impl = sum(mid(p) for p in cell) / len(cell)
    act = 100 * sum(p["yes"] for p in cell) / len(cell)
    check("ERRATUM A: pro 40-49 mid n", len(cell), 42)
    check("ERRATUM A: pro 40-49 mid implied", round(impl, 1), 45.5)
    check("ERRATUM A: pro 40-49 mid actual", round(act, 1), 19.0)
    P(f"  40-49 midpoint cell: n={len(cell)} implied={impl:.1f}% actual={act:.1f}%  (manuscript says n=76, 46.8% — freeze says n=42, 45.5%)")
    b2029 = [p for p in pro if 20 <= p["p"] <= 29]
    econ = [e for e in (trade_economics(p, "no") for p in b2029) if e is not None]
    check("ERRATUM B: pro 20-29 NO n", len(econ), 56)
    check("ERRATUM B: pro 20-29 NO mean", round(sum(econ) / len(econ), 2), 6.98)
    P(f"  20-29 BUY NO: n={len(econ)} mean={sum(econ)/len(econ):+.2f}c  (manuscript says +6.75, n=55 — freeze says +6.98, n=56)")
    g = boot_mean_ci(pro, lambda r: r["yes"] * 100 - float(r["p"]))
    check("pro display gap", round(g[0], 2), -5.86)
    P(f"  display global gap: {fmt_ci(g[:3])} pp (n=937)")
    gm_ = boot_mean_ci(pts, lambda r: r["yes"] * 100 - mid(r))
    check("pro midpoint gap", round(gm_[0], 2), -7.50)
    P(f"  midpoint global gap: {fmt_ci(gm_[:3])} pp (n=583)")

    n_ok = sum(1 for c in CHECKS if c[3]); n_bad = sum(1 for c in CHECKS if not c[3])
    P(f"\nVERIFICATION: {n_ok} checks passed, {n_bad} failed.")
    for c in CHECKS:
        if not c[3]:
            P(f"  FAIL: {c[0]} got {c[1]} want {c[2]}")
    out["verify"] = {"passed": n_ok, "failed": n_bad,
                     "failures": [c[0] for c in CHECKS if not c[3]]}

    # ================= PART 2: NEW ROBUSTNESS =================
    P("\n" + "=" * 78); P("PART 2 — NEW (POST-FREEZE) ROBUSTNESS ANALYSES"); P("=" * 78)

    # A. Spread-stratified midpoint calibration (retro)
    P("\n[A] Retro midpoint calibration by entry spread (a - b):")
    P(f"  {'stratum':>9} {'n':>6} {'clus':>6} {'mean mid':>8} | global gap pp [95% CI] | mid-curve (30<=mid<60) implied->actual, gap [CI]")
    strata = (("1-2c", 1, 2), ("3-5c", 3, 5), ("6-10c", 6, 10), ("11-20c", 11, 20), (">=21c", 21, 10**9))
    out["new"]["spread_strata"] = []
    for name, lo_s, hi_s in strata:
        sub = [p for p in ts if lo_s <= (p["a"] - p["b"]) <= hi_s]
        if not sub: continue
        ncl = len({p['cluster'] for p in sub})
        g = boot_mean_ci(sub, lambda r: r["yes"] * 100 - mid(r))
        mc = [p for p in sub if 30 <= mid(p) < 60]
        row = {"stratum": name, "n": len(sub), "clusters": ncl, "mean_mid": round(sum(mid(p) for p in sub)/len(sub), 1),
               "gap": round(g[0], 2), "gap_ci": [round(g[1], 2), round(g[2], 2)], "midcurve": None}
        line = f"  {name:>9} {len(sub):>6} {ncl:>6} {row['mean_mid']:>7.1f}c | {g[0]:+6.2f} [{g[1]:+6.2f}, {g[2]:+6.2f}]"
        if len(mc) >= 100:
            gi = sum(mid(p) for p in mc) / len(mc)
            ga = 100 * sum(p["yes"] for p in mc) / len(mc)
            gg = boot_mean_ci(mc, lambda r: r["yes"] * 100 - mid(r))
            row["midcurve"] = {"n": len(mc), "implied": round(gi, 1), "actual": round(ga, 1),
                               "gap": round(gg[0], 2), "gap_ci": [round(gg[1], 2), round(gg[2], 2)]}
            line += f" | mid-curve n={len(mc):>5}: {gi:.1f}%->{ga:.1f}%, gap {gg[0]:+.2f} [{gg[1]:+.2f}, {gg[2]:+.2f}]"
        P(line)
        out["new"]["spread_strata"].append(row)
    # mechanical-centering diagnostic
    P("\n  Composition: share of each midpoint bucket with spread >= 21c:")
    comp = {}
    for b0 in range(0, 100, 10):
        bp = [p for p in ts if b0 <= mid(p) <= b0 + 9.99]
        wide = sum(1 for p in bp if (p["a"] - p["b"]) >= 21)
        comp[f"{b0}-{b0+9}"] = round(100 * wide / len(bp), 1)
    P("  " + ", ".join(f"{k}: {v}%" for k, v in comp.items()))
    out["new"]["widebook_share_by_midbucket"] = comp

    # B. Paired display-minus-midpoint gap difference (retro)
    P("\n[B] Retro paired gap difference, gap_display - gap_midpoint = mean(mid - display):")
    w = boot_mean_ci(ts, lambda r: mid(r) - float(r["p"]))
    P(f"  all two-sided rows (n={w[3]}, clus={w[4]}): {fmt_ci(w[:3])} pp")
    expl = [p for p in ts if p["p_explicit"]]
    w2 = boot_mean_ci(expl, lambda r: mid(r) - float(r["p"])) if expl else None
    P(f"  rows with explicit candle display price (n={len(expl)}): " + (fmt_ci(w2[:3]) + " pp" if w2 else "n/a"))
    P(f"  note: rows without explicit display fall back to display:=round(mid), so the all-rows contrast is diluted toward zero; the explicit-price subset is the honest display-layer contrast.")
    out["new"]["paired_wedge_retro"] = {"all": [round(x, 3) for x in w[:3]], "n_all": w[3],
                                        "explicit": ([round(x, 3) for x in w2[:3]] if w2 else None),
                                        "n_explicit": len(expl)}
    # upper-tail paired contrast on identical rows (display-bucketed 90s)
    tail = [p for p in ts if 90 <= p["p"] <= 99]
    gi = sum(p["p"] for p in tail) / len(tail); ga = 100 * sum(p["yes"] for p in tail) / len(tail)
    gm_t = sum(mid(p) for p in tail) / len(tail)
    P(f"  90-99 display-bucket rows (n={len(tail)}): mean display {gi:.1f}%, mean mid {gm_t:.1f}%, actual {ga:.1f}%")

    # C. Prospective apples-to-apples (583 two-sided)
    P("\n[C] Prospective display calibration restricted to the 583 two-sided markets:")
    gd = boot_mean_ci(pts, lambda r: r["yes"] * 100 - float(r["p"]))
    P(f"  display gap on 583: {fmt_ci(gd[:3])} pp   (vs midpoint gap on same 583: {fmt_ci(gm_[:3])} pp)")
    wp = boot_mean_ci(pts, lambda r: mid(r) - float(r["p"]))
    P(f"  paired wedge mean(mid - display) on 583: {fmt_ci(wp[:3])} pp")
    tail = [p for p in pts if 90 <= p["p"] <= 99]
    if tail:
        gi = sum(p["p"] for p in tail) / len(tail); ga = 100 * sum(p["yes"] for p in tail) / len(tail)
        P(f"  90-99 display bucket within two-sided (n={len(tail)}): implied {gi:.1f}% actual {ga:.1f}%")
        a_ci = boot_mean_ci(tail, lambda r: r["yes"] * 100.0)
        P(f"    actual-rate 95% CI: [{a_ci[1]:.1f}, {a_ci[2]:.1f}]")
    out["new"]["prospective_583"] = {"display_gap": [round(x, 2) for x in gd[:3]],
                                     "midpoint_gap": [round(x, 2) for x in gm_[:3]],
                                     "paired_wedge": [round(x, 2) for x in wp[:3]]}

    json.dump(out, open(f"{OUTD}/robustness.json", "w"), indent=2)
    P(f"\nartifacts: {OUTD}/robustness_report.txt , robustness.json")
    R.close()

if __name__ == "__main__":
    sys.exit(main())
