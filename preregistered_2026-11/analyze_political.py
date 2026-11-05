"""Pre-specified political compression test (preregistered_2026-11/PLAN_political_compression_test.md, commit 8e46d8e).
Usage: python analyze_political.py ENTRIES_JSONL OUT_JSON"""
import json, sys
import numpy as np, pandas as pd
ENT, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(20260824); DRAWS = 2000
df = pd.DataFrame([json.loads(l) for l in open(ENT)])
df = df[df.status == "ok"].copy()
df["y"] = (df.result == "yes").astype(float)
df["mid"] = (df.a + df.b) / 2; df["spread"] = df.a - df.b
df["ev_n"] = df.groupby("event")["ticker"].transform("count")

def stats(sub, col):
    sub = sub[sub[col].notna()]
    x, y = sub[col].to_numpy() / 100, sub.y.to_numpy()
    g = pd.DataFrame({"e": sub.event.to_numpy(), "n": 1.0, "x": x, "y": y, "xx": x * x, "xy": x * y, "gap": y * 100 - x * 100}).groupby("e").sum().to_numpy()
    def est(t):
        n, sx, sy, sxx, sxy, sg = (t[..., i] for i in range(6))
        slope = (sxy - sx * sy / n) / (sxx - sx * sx / n)
        return slope, sy / n - slope * sx / n, sg / n
    s0, i0, g0 = est(g.sum(0)); k = len(g)
    bs = np.empty((DRAWS, 3))
    for j in range(0, DRAWS, 200):
        m = min(200, DRAWS - j); idx = rng.integers(0, k, size=(m, k)); t = g[idx].sum(1)
        bs[j:j + m] = np.column_stack(est(t))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"n": int(len(sub)), "events": int(k), "slope": round(float(s0), 3), "slope_ci": [round(float(lo[0]), 3), round(float(hi[0]), 3)],
            "intercept": round(float(i0), 3), "gap": round(float(g0), 2), "gap_ci": [round(float(lo[2]), 2), round(float(hi[2]), 2)]}

S = {"P0_all": df, "P1_clean_spread<=50_events1-2": df[(df.spread <= 50) & (df.ev_n <= 2)], "P2_spread<=50": df[df.spread <= 50],
     "P3_spread>=51": df[df.spread >= 51], "P4_spread<=4": df[df.spread <= 4]}
R = {"universe": {"markets": int(len(df)), "events": int(df.event.nunique()),
                  "share_spread>=51": round(float((df.spread >= 51).mean()), 3), "share_events3plus": round(float((df.ev_n >= 3).mean()), 3)},
     "midpoint": {k: stats(v, "mid") for k, v in S.items()},
     "trade_price": {k: stats(v, "trade") for k, v in (("P0_all", df), ("P1_clean_spread<=50_events1-2", S["P1_clean_spread<=50_events1-2"]))}}
p0, p1 = R["midpoint"]["P0_all"], R["midpoint"]["P1_clean_spread<=50_events1-2"]
replicated = p0["slope_ci"][0] > 1
thin = p1["n"] < 200 or p1["events"] < 30
if not replicated: verdict = "not replicated (P0 slope CI not entirely above 1)"
elif thin: verdict = "replicated; P1 too thin for a decision (descriptive only)"
elif p1["slope_ci"][0] > 1: verdict = "replicated; compression SURVIVES in tight books"
elif p1["slope_ci"][0] <= 1 <= p1["slope_ci"][1] and abs(p1["slope"] - 1) < abs(p0["slope"] - 1): verdict = "replicated; ARTIFACT-CONSISTENT (P1 CI includes 1 and is closer to 1)"
else: verdict = "inconclusive"
R["verdict_preregistered"] = verdict
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps(R, indent=1))
