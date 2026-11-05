#!/usr/bin/env python3
"""Supplementary Figure S1 (descriptive): entry-spread distributions in both studies. Swan (2026).
Usage: python figS1.py EXPLORATORY_JSON OUT_PREFIX"""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); OUT = sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
info = {}
for ax, (k, title) in zip(axes, (("study1", "A. Study 1 (57,808 markets)"), ("study2", "B. Study 2 (6,922 political markets)"))):
    c = {int(s): n for s, n in R[k]["E_spread_counts"].items()}; tot = sum(c.values())
    xs = list(range(1, 100)); ys = [100 * c.get(x, 0) / tot for x in xs]
    ax.bar(xs, ys, width=0.85, color="0.35", linewidth=0)
    ax.axvline(50.5, color="0.15", lw=0.9, ls=(0, (4, 3)))
    ax.text(52, max(ys) * 0.92, "50¢ cutoff", fontsize=7.5, color="0.2", ha="left")
    ax.set_title(title, fontsize=9, loc="left"); ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 10))
    ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_xlabel("Entry spread (¢)")
    info[k] = {"n": tot, "share_le4": round(sum(c.get(x, 0) for x in range(1, 5)) / tot, 3), "share_5_50": round(sum(c.get(x, 0) for x in range(5, 51)) / tot, 3),
               "share_51_70": round(sum(c.get(x, 0) for x in range(51, 71)) / tot, 3), "share_ge71": round(sum(c.get(x, 0) for x in range(71, 100)) / tot, 3),
               "mode": max(xs, key=lambda x: c.get(x, 0)), "top_wide": sorted(range(51, 100), key=lambda x: -c.get(x, 0))[:5]}
axes[0].set_ylabel("Share of markets (%)")
fig.tight_layout(w_pad=1.2)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
from PIL import Image
w, h = Image.open(f"{OUT}.png").size
info["w"], info["h"] = w, h
json.dump(info, open(f"{OUT}.json", "w"), indent=1); print(json.dumps(info))
