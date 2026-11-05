#!/usr/bin/env python3
"""
Book-width decomposition of calibration and taker economics (post-freeze, exploratory).
Swan (2026). Reads frozen logs/retro_pairs.jsonl (+ prospective snapshots/resolutions),
same loader and fee rules as tools/gate_analysis.py.

Usage: python width_decomposition.py [LOGS_DIR] [OUT_DIR]
"""
import json, math, sys
import numpy as np, pandas as pd

LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
rng = np.random.default_rng(20260824)
DRAWS = 2000

def load_retro():
    rows = []
    for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("r") not in ("yes", "no"):
            continue
        b, a = int(r.get("b") or 0), int(r.get("a") or 100)
        p = r.get("p") or round((b + a) / 2)
        rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "",
                     "b": b, "a": a, "p": int(p), "y": 1 if r["r"] == "yes" else 0})
    return pd.DataFrame(rows)

def fee(cost, mult):
    q = cost / 100.0
    return math.ceil(mult * q * (1 - q) * 100)

df = load_retro()
df = df[(df.b > 0) & (df.a < 100) & (df.a > df.b)].copy()
assert len(df) == 57808
df["mid"] = (df.a + df.b) / 2
df["spread"] = df.a - df.b
df["gap"] = df.y * 100 - df.mid
df["ev_n"] = df.groupby("event")["t"].transform("count")
df["dbucket"] = np.minimum(df.p // 10, 9)  # display-price bucket, as in Table 3
for m in (0.0, 0.07):
    df[f"no_{m}"] = [(100 if y == 0 else 0) - (100 - b) - fee(100 - b, m) for y, b in zip(df.y, df.b)]
    df[f"yes_{m}"] = [(100 if y == 1 else 0) - a - fee(a, m) for y, a in zip(df.y, df.a)]

def boot(sub, col, draws=DRAWS):
    g = sub.groupby("event")[col].agg(["sum", "count"])
    S, N = g["sum"].to_numpy(dtype=float), g["count"].to_numpy()
    k = len(S)
    if k == 0:
        return [np.nan] * 3 + [0, 0]
    out = np.empty(draws)
    for i in range(0, draws, 400):
        m = min(400, draws - i)
        idx = rng.integers(0, k, size=(m, k))
        out[i:i + m] = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [S.sum() / N.sum(), lo, hi, len(sub), k]

def slope_int(sub):
    x = sub.mid.to_numpy() / 100; y = sub.y.to_numpy().astype(float)
    xm, ym = x.mean(), y.mean(); s = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return s, ym - s * xm

R, L = {}, []
def P(s=""):
    print(s); L.append(s)
f = lambda t: f"{t[0]:+7.2f} [{t[1]:+7.2f}, {t[2]:+7.2f}] n={t[3]:>6} clus={t[4]:>5}"

groups = {"<=4c": df.spread <= 4, "5-20c": (df.spread >= 5) & (df.spread <= 20),
          "21-50c": (df.spread >= 21) & (df.spread <= 50), ">=51c": df.spread >= 51,
          "<=50c": df.spread <= 50, "all": df.spread > 0}

P("[1] Spread distribution (retrospective, n=57,808)")
R["spread_dist"] = {}
for lab in ("<=4c", "5-20c", "21-50c", ">=51c"):
    m = groups[lab]; R["spread_dist"][lab] = {"n": int(m.sum()), "share": round(float(m.mean()), 4)}
    P(f"  {lab:>6}: n={int(m.sum()):>6}  share {100*m.mean():5.1f}%")
P(f"  near-empty (bid<=5, ask>=90): {int(((df.b<=5)&(df.a>=90)).sum())}  share {100*((df.b<=5)&(df.a>=90)).mean():.1f}%")
P(f"  spread >70c: {int((df.spread>70).sum())} share {100*(df.spread>70).mean():.1f}%")

P("\n[2] Global midpoint calibration by spread group")
R["global"] = {}
for lab, m in groups.items():
    sub = df[m]; g = boot(sub, "gap"); s, i = slope_int(sub)
    br = float(((sub.mid / 100 - sub.y) ** 2).mean())
    R["global"][lab] = {"gap": [round(v, 2) for v in g[:3]], "n": g[3], "clusters": g[4],
                        "slope": round(s, 3), "intercept": round(i, 3), "brier": round(br, 4)}
    P(f"  {lab:>6}: gap {f(g)}  slope {s:.3f}  intercept {i:+.3f}  Brier {br:.4f}")

P("\n[3] Midpoint gap by spread group x event size (markets per event in sample)")
R["by_eventsize"] = {}
for lab in ("<=4c", "5-20c", "21-50c", ">=51c"):
    row = {}
    for elab, em in (("1-2", df.ev_n <= 2), ("3+", df.ev_n >= 3)):
        g = boot(df[groups[lab] & em], "gap"); row[elab] = {"gap": [round(v, 2) for v in g[:3]], "n": g[3], "clusters": g[4]}
    R["by_eventsize"][lab] = row
    P(f"  {lab:>6}: 1-2 mkts {f([row['1-2']['gap'][0], row['1-2']['gap'][1], row['1-2']['gap'][2], row['1-2']['n'], row['1-2']['clusters']])}"
      f" | 3+ mkts {f([row['3+']['gap'][0], row['3+']['gap'][1], row['3+']['gap'][2], row['3+']['n'], row['3+']['clusters']])}")

P("\n[4] Pooled taker returns (cents/contract) by spread group, fee 0 and 0.07")
R["taker_pooled"] = {}
for lab, m in groups.items():
    sub = df[m]; row = {}
    for side in ("no", "yes"):
        for mult in (0.0, 0.07):
            g = boot(sub, f"{side}_{mult}"); row[f"{side}_{mult}"] = [round(v, 2) for v in g[:3]]
    row["n"] = int(len(sub)); R["taker_pooled"][lab] = row
    P(f"  {lab:>6}: NO fee0 {row['no_0.0']}  NO fee.07 {row['no_0.07']} | YES fee0 {row['yes_0.0']}  YES fee.07 {row['yes_0.07']}")

P("\n[5] Gate check within narrow books (<=20c), display-price buckets, fee 0.07 (exploratory subgroup)")
R["gate_narrow"] = []
nar = df[df.spread <= 20]
for side in ("no", "yes"):
    for bk in range(10):
        sub = nar[nar.dbucket == bk]
        if len(sub) < 10:
            continue
        g = boot(sub, f"{side}_0.07")
        passes = g[3] >= 200 and g[4] >= 30 and g[1] > 0
        R["gate_narrow"].append({"side": side, "bucket": f"{10*bk}-{10*bk+9}", "mean": round(g[0], 2),
                                 "ci": [round(g[1], 2), round(g[2], 2)], "n": g[3], "clusters": g[4], "passes": bool(passes)})
        P(f"  BUY {side.upper():>3} {10*bk:>2}-{10*bk+9:<2}: {f(g)}{'  *** PASSES GATE ***' if passes else ''}")
P(f"  any narrow-book cell passing the gate: {any(x['passes'] for x in R['gate_narrow'])}")

json.dump(R, open(f"{OUT}/width_decomposition.json", "w"), indent=1)
open(f"{OUT}/width_decomposition.txt", "w").write("\n".join(L) + "\n")
print(f"\nartifacts: {OUT}/width_decomposition.txt, .json")
