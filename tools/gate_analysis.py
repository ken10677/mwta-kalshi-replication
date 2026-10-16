"""
THE RE-AUTOMATION GATE ANALYSIS (2026-08-24)

The constitutional question, written into governance on 2026-05-21 and
2026-06-24: does any price bucket show a bias that clears fees and spread
at 95% confidence, with CLUSTER-AWARE statistics, on >=200 observations?

Design (the honesty features are the whole point):

1. JOIN: earliest snapshot per market (logs/market_snapshots.jsonl) x
   binary resolution (logs/resolutions.jsonl). The earliest snapshot is the
   most conservative: it's the price you could actually have acted on,
   furthest from resolution.

2. CLUSTERS: markets that resolve on the same underlying event are one
   observation, not many (one CPI print settles a whole ladder). Cluster
   key = ticker base with the trailing threshold segment (-T.../-B...)
   stripped, e.g. KXCPIYOY-26JUL-T4.1 -> KXCPIYOY-26JUL. All statistics
   use a cluster bootstrap (resample CLUSTERS with replacement, 10,000
   draws, fixed seed): if the edge only exists because one hot month
   settled 100 correlated markets the same way, the CI will say so.

3. REALIZABLE ECONOMICS, not calibration gaps: the observed bias is
   "YES overpriced" -> the candidate trade is BUY NO. We price entry at
   the NO ask from the snapshot (100 - yes_bid) — what a taker actually
   pays — require a two-sided book at snapshot time, and charge Kalshi's
   real taker fee (signals.fees.calculate_fee_cents, ceil'd). Settlement
   itself is fee-free. The mirror trade (BUY YES at the ask) is computed
   for every bucket too; if the story is real, it should be negative.

4. GATE: a bucket PASSES only if raw n >= 200 AND clusters >= 30 AND the
   cluster-bootstrap 95% CI lower bound of net profit per contract > 0.

Output: console report + logs/gate_analysis.json artifact.
Run: python tools/gate_analysis.py
"""

import sys, os, json, math, random, re
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals.fees import calculate_fee_cents

SNAPSHOTS = "logs/market_snapshots.jsonl"
RESOLUTIONS = "logs/resolutions.jsonl"
OUT = "logs/gate_analysis.json"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260824
MIN_N = 200
MIN_CLUSTERS = 30

_THRESH_SUFFIX = re.compile(r"-[TB][\d.\-]+$")


def cluster_key(ticker: str) -> str:
    """KXCPIYOY-26JUL-T4.1 -> KXCPIYOY-26JUL (one data print = one cluster)."""
    return _THRESH_SUFFIX.sub("", ticker)


def load_pairs():
    res = {}
    for line in open(RESOLUTIONS, encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") in ("yes", "no"):
            res[r["t"]] = r["r"]
    first = {}
    for line in open(SNAPSHOTS, encoding="utf-8"):
        s = json.loads(line)
        if s["t"] not in first:
            first[s["t"]] = s
    pairs = []
    for t, s in first.items():
        if t in res:
            pairs.append({
                "t": t, "p": s["p"], "b": s.get("b") or 0, "a": s.get("a") or 100,
                "d": s.get("d"), "yes": 1 if res[t] == "yes" else 0,
                "cluster": cluster_key(t),
                "tth": s.get("htc"),  # hours to close at entry
            })
    return pairs


def load_pairs_retro():
    """Retrospective stratum from candlestick reconstruction (Track B).

    Field mapping: b/a are the earliest two-sided daily candle's close
    quotes; p (display) is the candle price close when present, else the
    book mid (so Stage-1 display calibration is approximate here — the
    retro stratum's authoritative stages are mid/ask/fee). Cluster is the
    exchange's event_ticker — strictly better than the prefix regex.
    """
    pairs = []
    for line in open("logs/retro_pairs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") not in ("yes", "no"):
            continue
        b, a = int(r.get("b") or 0), int(r.get("a") or 100)
        p = r.get("p") or round((b + a) / 2)
        tth = None
        try:
            close_dt = datetime.fromisoformat(str(r["close"]).replace("Z", "+00:00"))
            if r.get("qts"):
                tth = (close_dt.timestamp() - float(r["qts"])) / 3600
        except Exception:
            pass
        pairs.append({
            "t": r["t"], "p": int(p), "b": b, "a": a,
            "d": (r.get("open") or "")[:10],
            "yes": 1 if r["r"] == "yes" else 0,
            "cluster": r.get("event") or cluster_key(r["t"]),
            "tth": tth,
        })
    return pairs


def fee_cents_mult(cost_cents: int, mult: float) -> int:
    """Kalshi-style fee: ceil(mult * P * (1-P) * 100). mult=0.07 is the
    standard taker coefficient; sensitivity analysis varies it because the
    exchange applies HIGHER multipliers to some premium categories (never
    lower for takers), making 0.07 a lower bound on taker fees."""
    p = cost_cents / 100.0
    return math.ceil(mult * p * (1 - p) * 100)


def trade_economics(pair, side, mult: float = 0.07):
    """
    Net profit in cents per contract, taker entry at snapshot quotes,
    hold to settlement. Returns None if the book wasn't two-sided.
    """
    b, a = pair["b"], pair["a"]
    if not (0 < b and a < 100 and a > b):
        return None
    if side == "no":
        cost = 100 - b                      # NO ask = 100 - yes_bid
        payout = 100 if pair["yes"] == 0 else 0
    else:
        cost = a                            # YES ask
        payout = 100 if pair["yes"] == 1 else 0
    return payout - cost - fee_cents_mult(int(cost), mult)


def cluster_bootstrap_ci(values_by_cluster, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """95% CI for the pooled mean, resampling clusters with replacement."""
    rng = random.Random(seed)
    clusters = list(values_by_cluster.values())
    if not clusters:
        return (0.0, 0.0, 0.0)
    means = []
    k = len(clusters)
    for _ in range(draws):
        pooled = []
        for _ in range(k):
            pooled.extend(clusters[rng.randrange(k)])
        means.append(sum(pooled) / len(pooled))
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[int(0.975 * draws)]
    all_vals = [v for c in clusters for v in c]
    return (sum(all_vals) / len(all_vals), lo, hi)


def cluster_bootstrap_stat(tuples_by_cluster, stat, draws=BOOTSTRAP_DRAWS,
                           seed=SEED):
    """95% CI for an arbitrary statistic over pooled per-cluster tuples.

    `stat` maps a list of tuples -> float. Used to bootstrap the
    calibration GAP mean(Y - p) directly (audit item: intervals must bound
    the gap, not merely the outcome rate), the Brier score, and the
    calibration slope/intercept.
    """
    rng = random.Random(seed)
    clusters = list(tuples_by_cluster.values())
    if not clusters:
        return (0.0, 0.0, 0.0)
    k = len(clusters)
    stats = []
    for _ in range(draws):
        pooled = []
        for _ in range(k):
            pooled.extend(clusters[rng.randrange(k)])
        stats.append(stat(pooled))
    stats.sort()
    all_vals = [t for c in clusters for t in c]
    return (stat(all_vals), stats[int(0.025 * draws)], stats[int(0.975 * draws)])


def _gap(tuples):
    """mean(Y*100 - p) over (y, p) tuples."""
    return sum(y * 100 - p for y, p in tuples) / len(tuples)


def _brier(tuples):
    """mean((p/100 - y)^2)."""
    return sum((p / 100 - y) ** 2 for y, p in tuples) / len(tuples)


def _cal_slope_intercept(tuples):
    """OLS of y on p/100: returns (slope, intercept). Perfect calibration:
    slope 1, intercept 0."""
    n = len(tuples)
    xs = [p / 100 for _, p in tuples]
    ys = [float(y) for y, _ in tuples]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (0.0, my)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (slope, my - slope * mx)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro", action="store_true",
                    help="analyze the retrospective candlestick stratum")
    args = ap.parse_args()
    pairs = load_pairs_retro() if args.retro else load_pairs()
    global OUT
    if args.retro:
        OUT = "logs/gate_analysis_retro.json"
    dates = sorted(p["d"] for p in pairs if p.get("d"))
    print("=" * 74)
    print("  RE-AUTOMATION GATE ANALYSIS" + ("  [RETRO STRATUM]" if args.retro else "  [PROSPECTIVE]"))
    print(f"  {len(pairs)} snapshot->outcome pairs | {dates[0]} .. {dates[-1]}")
    print(f"  clusters (events): {len({p['cluster'] for p in pairs})}")
    print(f"  bootstrap: {BOOTSTRAP_DRAWS} cluster resamples, seed {SEED}")
    print("=" * 74)

    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "pairs": len(pairs), "date_range": [dates[0], dates[-1]],
              "gate": {"min_n": MIN_N, "min_clusters": MIN_CLUSTERS},
              "buckets": []}

    # --- Calibration with cluster CIs (the science table) ---
    print("\nCALIBRATION (cluster-bootstrap 95% CI on actual YES rate)")
    print(f"{'bucket':>8} {'n':>5} {'clus':>5} {'implied':>8} {'actual':>7} {'95% CI':>16}")
    for b0 in range(0, 100, 10):
        bp = [p for p in pairs if b0 <= p["p"] <= b0 + 9]
        if len(bp) < 10:
            continue
        by_cl = defaultdict(list)
        for p in bp:
            by_cl[p["cluster"]].append(p["yes"] * 100)
        implied = sum(p["p"] for p in bp) / len(bp)
        mean, lo, hi = cluster_bootstrap_ci(by_cl)
        print(f"{b0:>3}-{b0+9:<4} {len(bp):>5} {len(by_cl):>5} {implied:>7.1f}% "
              f"{mean:>6.1f}% [{lo:>5.1f}, {hi:>5.1f}]")

    # --- Calibration at BOOK MID (two-sided only): the staleness test ---
    # Deep-research positioning (2026-08-24): the paper's staged decomposition
    # is display -> mid -> ask -> fee. If miscalibration shrinks at book mid,
    # the displayed-price anomaly is substantially a stale/reference-price
    # artifact; whatever remains at mid is book-level bias, and the ask/fee
    # stage (below) shows whether even that is capturable.
    print("\nCALIBRATION AT BOOK MID (two-sided books only — staleness diagnostic)")
    print(f"{'bucket':>8} {'n':>5} {'clus':>5} {'implied':>8} {'actual':>7} {'95% CI':>16}")
    report["mid_calibration"] = []
    two_sided = [p for p in pairs if 0 < p["b"] and p["a"] < 100 and p["a"] > p["b"]]
    for b0 in range(0, 100, 10):
        bp = [p for p in two_sided if b0 <= (p["b"] + p["a"]) / 2 <= b0 + 9.99]
        if len(bp) < 10:
            continue
        by_cl = defaultdict(list)
        for p in bp:
            by_cl[p["cluster"]].append(p["yes"] * 100)
        implied = sum((p["b"] + p["a"]) / 2 for p in bp) / len(bp)
        mean, lo, hi = cluster_bootstrap_ci(by_cl)
        print(f"{b0:>3}-{b0+9:<4} {len(bp):>5} {len(by_cl):>5} {implied:>7.1f}% "
              f"{mean:>6.1f}% [{lo:>5.1f}, {hi:>5.1f}]")
        report["mid_calibration"].append({
            "bucket": f"{b0}-{b0+9}", "n": len(bp), "clusters": len(by_cl),
            "implied_pct": round(implied, 1), "actual_pct": round(mean, 1),
            "ci95": [round(lo, 1), round(hi, 1)],
        })

    # --- Tradeable economics (the money table) ---
    verdicts = []
    for side, label in (("no", "BUY NO"), ("yes", "BUY YES")):
        print(f"\nSTRATEGY: {label} at snapshot ask, hold to settlement, taker fee")
        print(f"{'bucket':>8} {'n':>5} {'clus':>5} {'avg P/L':>9} {'95% CI':>18} "
              f"{'avg cost':>9} {'ROI':>7}  gate")
        for b0 in range(0, 100, 10):
            bp = [p for p in pairs if b0 <= p["p"] <= b0 + 9]
            econ = [(p, trade_economics(p, side)) for p in bp]
            econ = [(p, e) for p, e in econ if e is not None]
            if len(econ) < 10:
                continue
            by_cl = defaultdict(list)
            cost_total = 0
            for p, e in econ:
                by_cl[p["cluster"]].append(e)
                cost_total += (100 - p["b"]) if side == "no" else p["a"]
            mean, lo, hi = cluster_bootstrap_ci(by_cl)
            avg_cost = cost_total / len(econ)
            roi = 100 * mean / avg_cost
            passes = (len(econ) >= MIN_N and len(by_cl) >= MIN_CLUSTERS and lo > 0)
            flag = "*** PASS ***" if passes else (
                "near-miss" if lo > 0 else "")
            print(f"{b0:>3}-{b0+9:<4} {len(econ):>5} {len(by_cl):>5} "
                  f"{mean:>+8.2f}c [{lo:>+7.2f}, {hi:>+6.2f}] {avg_cost:>8.1f}c "
                  f"{roi:>+6.2f}%  {flag}")
            report["buckets"].append({
                "side": side, "bucket": f"{b0}-{b0+9}", "n": len(econ),
                "clusters": len(by_cl), "mean_pl_cents": round(mean, 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "avg_cost_cents": round(avg_cost, 1),
                "roi_pct": round(roi, 3), "passes_gate": passes,
            })
            if passes:
                verdicts.append((label, f"{b0}-{b0+9}", mean, lo, hi, len(econ), len(by_cl)))

    # ================= AUDIT-DRIVEN EXHIBITS (v2, 2026-08-25) =================
    two_sided = [p for p in pairs if 0 < p["b"] and p["a"] < 100 and p["a"] > p["b"]]

    def _mid(p):
        return (p["b"] + p["a"]) / 2

    # --- Global forecast-evaluation metrics with direct gap bootstrap ---
    print("\nGLOBAL FORECAST EVALUATION (cluster bootstrap; gap = mean(Y*100 - p))")
    report["global"] = {}
    for label, sample, price_of in (
            ("display", pairs, lambda p: float(p["p"])),
            ("midpoint", two_sided, _mid)):
        by_cl = defaultdict(list)
        for p in sample:
            by_cl[p["cluster"]].append((p["yes"], price_of(p)))
        gap, glo, ghi = cluster_bootstrap_stat(by_cl, _gap)
        br, blo, bhi = cluster_bootstrap_stat(by_cl, _brier)
        sl, sllo, slhi = cluster_bootstrap_stat(by_cl, lambda t: _cal_slope_intercept(t)[0])
        ic, iclo, ichi = cluster_bootstrap_stat(by_cl, lambda t: _cal_slope_intercept(t)[1])
        print(f"  {label:>8}: n={len(sample)}  gap={gap:+.2f}pp [{glo:+.2f}, {ghi:+.2f}]  "
              f"Brier={br:.4f} [{blo:.4f}, {bhi:.4f}]")
        print(f"  {'':>8}  calib slope={sl:.3f} [{sllo:.3f}, {slhi:.3f}]  "
              f"intercept={ic:.3f} [{iclo:.3f}, {ichi:.3f}]  (perfect: 1, 0)")
        report["global"][label] = {
            "n": len(sample), "gap_pp": round(gap, 3), "gap_ci": [round(glo, 3), round(ghi, 3)],
            "brier": round(br, 5), "brier_ci": [round(blo, 5), round(bhi, 5)],
            "slope": round(sl, 4), "slope_ci": [round(sllo, 4), round(slhi, 4)],
            "intercept": round(ic, 4), "intercept_ci": [round(iclo, 4), round(ichi, 4)],
        }

    # --- Horizon strata (midpoint gap by time-to-close at entry) ---
    print("\nHORIZON STRATA (midpoint calibration gap by time-to-close at entry)")
    report["horizon"] = []
    strata = (("<24h", 0, 24), ("24h-7d", 24, 168), (">=7d", 168, 10**9))
    for name, lo_h, hi_h in strata:
        sub = [p for p in two_sided if p.get("tth") is not None and lo_h <= p["tth"] < hi_h]
        if len(sub) < 30:
            print(f"  {name:>7}: n={len(sub)} (insufficient)")
            continue
        by_cl = defaultdict(list)
        for p in sub:
            by_cl[p["cluster"]].append((p["yes"], _mid(p)))
        gap, glo, ghi = cluster_bootstrap_stat(by_cl, _gap)
        print(f"  {name:>7}: n={len(sub):>6} clus={len(by_cl):>5}  "
              f"gap={gap:+.2f}pp [{glo:+.2f}, {ghi:+.2f}]")
        report["horizon"].append({"stratum": name, "n": len(sub),
                                  "clusters": len(by_cl), "gap_pp": round(gap, 2),
                                  "ci": [round(glo, 2), round(ghi, 2)]})

    # --- Fee sensitivity: pooled taker EV under fee multipliers ---
    # Kalshi applies HIGHER multipliers to premium categories, never lower
    # for takers, so 0.07 lower-bounds taker fees; 0.00 bounds the frictionless
    # case. Pooled (all buckets) = multiplicity-free global test.
    print("\nFEE SENSITIVITY (pooled taker EV, cents/contract; 2,000 cluster draws)")
    report["fee_sensitivity"] = []
    for mult in (0.0, 0.035, 0.07, 0.14):
        row = {"mult": mult}
        line = f"  mult={mult:.3f}: "
        for side in ("no", "yes"):
            by_cl = defaultdict(list)
            for p in two_sided:
                e = trade_economics(p, side, mult)
                if e is not None:
                    by_cl[p["cluster"]].append(e)
            mean, lo, hi = cluster_bootstrap_ci(by_cl, draws=2000)
            line += f"BUY {side.upper()}: {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]   "
            row[f"pooled_{side}"] = {"mean": round(mean, 3),
                                     "ci": [round(lo, 3), round(hi, 3)]}
        print(line)
        report["fee_sensitivity"].append(row)

    # --- Cross-stratum overlap (audit: never sum overlapping samples) ---
    try:
        other = load_pairs() if args.retro else load_pairs_retro()
        overlap = {p["t"] for p in pairs} & {p["t"] for p in other}
        print(f"\nCROSS-STRATUM OVERLAP: {len(overlap)} tickers appear in both "
              f"strata (analyzed separately; counts never summed)")
        report["cross_stratum_overlap_tickers"] = len(overlap)
    except Exception as e:
        print(f"\nCROSS-STRATUM OVERLAP: unavailable ({e})")

    print("\n" + "=" * 74)
    if verdicts:
        print("  GATE VERDICT: PASS")
        for label, bucket, mean, lo, hi, n, cl in verdicts:
            print(f"    {label} {bucket}c: {mean:+.2f}c/contract "
                  f"[{lo:+.2f}, {hi:+.2f}] on n={n}, clusters={cl}")
        print("  Caveats: single macro regime (Jul-Aug 2026); quote depth unknown;")
        print("  passing here authorizes CONSIDERING small live tests, not scaling.")
    else:
        print("  GATE VERDICT: NO BUCKET PASSES")
        print("  No strategy clears fees+spread at 95% cluster-aware confidence.")
    print("=" * 74)
    report["verdict_pass"] = [
        {"strategy": v[0], "bucket": v[1], "mean": round(v[2], 3),
         "ci": [round(v[3], 3), round(v[4], 3)], "n": v[5], "clusters": v[6]}
        for v in verdicts]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nartifact: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
