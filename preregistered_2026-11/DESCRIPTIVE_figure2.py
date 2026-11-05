#!/usr/bin/env python3
"""Figure 2 (descriptive): reliability of entry midpoints in pre-registered political sample.
Panels use the pre-specified subsets P0 (all), P2 (spread <= 50c), P3 (spread >= 51c). Swan (2026).
Usage: python figure2.py ENTRIES_JSONL OUT_PREFIX"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ENT, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(20260824)
d = pd.DataFrame([json.loads(l) for l in open(ENT)]); d = d[d.status == "ok"].copy()
d["y"] = (d.result == "yes") * 100.0; d["mid"] = (d.a + d.b) / 2; d["spread"] = d.a - d.b
d["mb"] = np.minimum((d.mid // 10).astype(int), 9)
P = {"P0": ("A. All political markets", d), "P2": ("B. Spread ≤ 50¢", d[d.spread <= 50]), "P3": ("C. Spread ≥ 51¢", d[d.spread >= 51])}
R = {}
for k, (title, s) in P.items():
    out = []
    for bk in range(10):
        c = s[s.mb == bk]
        if len(c) < 30: continue
        g = c.groupby("event").y.agg(["sum", "count"]); S, N = g["sum"].to_numpy(), g["count"].to_numpy(); m = len(S)
        idx = rng.integers(0, m, size=(2000, m)); b = S[idx].sum(1) / N[idx].sum(1)
        out.append({"bucket": bk, "n": int(len(c)), "implied": round(float(c.mid.mean()), 2), "yes": round(float(S.sum() / N.sum()), 2),
                    "lo": round(float(np.percentile(b, 2.5)), 2), "hi": round(float(np.percentile(b, 97.5)), 2)})
    R[k] = {"title": title, "n": int(len(s)), "share": round(len(s) / len(d), 3), "buckets": out}
json.dump(R, open(OUT + ".json", "w"), indent=1)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), sharey=True)
res = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
for ax, k in zip(axes, ("P0", "P2", "P3")):
    r = R[k]
    ax.plot([0, 100], [0, 100], color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for p in r["buckets"]:
        ax.plot([p["implied"]] * 2, [p["lo"], p["hi"]], color="0.15", lw=0.9, zorder=2)
        ax.scatter(p["implied"], p["yes"], s=8 + p["n"] / 15, color="0.15", edgecolor="white", lw=0.6, zorder=3)
    lab = f"n = {r['n']:,} ({100*r['share']:.1f}%)"
    if res:
        key = {"P0": "P0_all", "P2": "P2_spread<=50", "P3": "P3_spread>=51"}[k]; sl = res["midpoint"][key]
        lab += f"\nslope {sl['slope']:.2f} [{sl['slope_ci'][0]:.2f}, {sl['slope_ci'][1]:.2f}]"
    ax.text(3, 96, lab, va="top", ha="left", fontsize=7.5, color="0.2")
    ax.set_title(r["title"], fontsize=9, loc="left"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20)); ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_xlabel("Mean midpoint in bucket (¢)")
axes[0].set_ylabel("Resolved YES (%)")
fig.tight_layout(w_pad=1.0)
for ext in ("pdf", "png", "tiff"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext != "pdf" else None, bbox_inches="tight", **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
print(json.dumps({k: [(b["bucket"], b["n"], b["implied"], b["yes"]) for b in v["buckets"]] for k, v in R.items()}))
