#!/usr/bin/env python3
"""Figure 1: reliability of order-book midpoints by book width and event structure.
Swan (2026). Reads width_event_structure.json (from width_event_structure.py).
Usage: python make_figure1.py [JSON] [OUT_PREFIX]"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
J = sys.argv[1] if len(sys.argv) > 1 else "width_event_structure.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "Figure_1"
R = json.load(open(J))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
panels = [("all", "A. All markets"),
          ("spread<=50", "B. Spread ≤ 50¢"),
          ("spread>=51 & 1-2 mkts", "C. Spread ≥ 51¢, event with 1–2 markets"),
          ("spread>=51 & 3+ mkts", "D. Spread ≥ 51¢, event with 3+ markets")]
comp = R["composition"]; gap = R["gap"]
fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.4), sharex=True, sharey=True)
for ax, (key, title) in zip(axes.flat, panels):
    pts = R["reliability"][key]
    ax.plot([0, 100], [0, 100], color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for p in pts:
        ax.plot([p["implied"]] * 2, [p["lo"], p["hi"]], color="0.15", lw=0.9, zorder=2)
        ax.scatter(p["implied"], p["yes"], s=8 + p["n"] / 60, color="0.15", edgecolor="white", lw=0.6, zorder=3)
    n, share = comp[key]["n"], comp[key]["share"]
    g = gap[key]
    m = lambda v: f"{v:+.2f}".replace("-", "\u2212")
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(3, 96, f"n = {n:,} ({100*share:.1f}%)\nmean gap {m(g['est'])} pp",
            va="top", ha="left", fontsize=7.5, color="0.2")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
    ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[1]: ax.set_xlabel("Mean midpoint in bucket (¢)")
for ax in axes[:, 0]: ax.set_ylabel("Resolved YES (%)")
fig.tight_layout(h_pad=1.6, w_pad=1.2)
for ext in ("pdf", "png", "tiff"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext != "pdf" else None, bbox_inches="tight",
                **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
print("wrote", OUT + ".pdf/.png/.tiff")
