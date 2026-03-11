#!/usr/bin/env python3
"""
fix_fig2_final.py — Clean Cost-Accuracy Pareto Frontier
Uses main plot + zoomed inset for the crowded hybrid top-left zone.
All 10 system families, all data points, no overlapping labels.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
OUT = os.path.join(FIGURES_DIR, "fig2_cost_accuracy_frontier.png")

# ─── DATA ────────────────────────────────────────────────────────────────────
POINTS = [
    # Encoders
    ("DeBERTa-base",   0.000,  0.9012, "Encoder"),
    ("DeBERTa-large",  0.000,  0.9012, "Encoder"),
    # GPT-4o
    ("GPT-4o P1",      0.204,  0.840,  "GPT-4o"),
    ("GPT-4o P2",      0.272,  0.829,  "GPT-4o"),
    ("GPT-4o P3",      0.375,  0.848,  "GPT-4o"),
    ("GPT-4o P4",      0.410,  0.855,  "GPT-4o"),
    ("GPT-4o P5",      3.520,  0.840,  "GPT-4o"),
    # Claude (costs from cost_summary.csv)
    ("Claude P1",      0.312,  0.874,  "Claude"),  # CSV: 0.312
    ("Claude P2",      0.399,  0.884,  "Claude"),  # FIX 2: was 0.410, CSV: 0.399
    ("Claude P3",      2.235,  0.885,  "Claude"),
    ("Claude P4",      2.800,  0.805,  "Claude"),
    # Llama
    ("Llama P1",       0.000,  0.746,  "Llama 3.3"),
    ("Llama P2",       0.000,  0.818,  "Llama 3.3"),
    ("Llama P3",       0.000,  0.779,  "Llama 3.3"),
    ("Llama P4",       0.000,  0.789,  "Llama 3.3"),
    # GPT-5
    ("GPT-5 P1",       14.83,  0.759,  "GPT-5"),
    ("GPT-5 P3",       17.54,  0.841,  "GPT-5"),
    ("GPT-5 P4",       17.54,  0.869,  "GPT-5"),
    # Hybrid v1 (costs from cost_summary.csv)
    ("v1 0.85",        0.009,  0.904,  "Hybrid v1"),   # CSV: 0.01187
    ("v1 0.90",        0.013,  0.9012, "Hybrid v1"),   # FIX 2: was 0.011, CSV: 0.01326
    ("v1 0.95",        0.023,  0.898,  "Hybrid v1"),   # CSV: 0.02291
    # Hybrid v2
    ("v2 0.85",        0.066,  0.902,  "Hybrid v2"),
    ("v2 0.90",        0.074,  0.9012, "Hybrid v2"),
    ("v2 0.95",        0.126,  0.896,  "Hybrid v2"),
    # Hybrid v3
    ("v3 0.85",        0.137,  0.900,  "Hybrid v3"),
    ("v3 0.90",        0.152,  0.899,  "Hybrid v3"),
    ("v3 0.95",        0.258,  0.894,  "Hybrid v3"),
    # Hybrid v4 BEST
    ("v4 0.85",        0.006,  0.9050, "Hybrid v4"),
    ("v4 0.90 BEST",   0.007,  0.9062, "Hybrid v4"),
    ("v4 0.95",        0.013,  0.9062, "Hybrid v4"),
    # Hybrid v5
    ("v5 Ensemble",    0.288,  0.895,  "Hybrid v5"),
]

STYLE = {
    "Encoder":   {"c": "#2166AC", "m": "D", "s": 100},
    "GPT-4o":    {"c": "#4DAC26", "m": "s", "s":  85},
    "Claude":    {"c": "#D01C8B", "m": "^", "s":  95},
    "Llama 3.3": {"c": "#F1A340", "m": "v", "s":  85},
    "GPT-5":     {"c": "#9970AB", "m": "X", "s":  95},
    "Hybrid v1": {"c": "#8DD3C7", "m": "o", "s":  75},
    "Hybrid v2": {"c": "#BEBADA", "m": "o", "s":  75},
    "Hybrid v3": {"c": "#80B1D3", "m": "o", "s":  75},
    "Hybrid v4": {"c": "#F28E2B", "m": "*", "s": 230},
    "Hybrid v5": {"c": "#33A02C", "m": "P", "s":  90},
}

def plot_points(ax, points, fontsize=8.5, show_labels=True, zorder_base=3):
    seen = set()
    for lbl, cost, acc, grp in points:
        st = STYLE[grp]
        leg = grp if grp not in seen else "_nolegend_"
        ax.scatter(cost, acc,
                   c=st["c"], marker=st["m"], s=st["s"],
                   label=leg, zorder=zorder_base + (2 if grp == "Hybrid v4" else 0),
                   edgecolors="white", linewidths=0.5, alpha=0.93)
        seen.add(grp)

def add_labels_main(ax):
    """Sparse labels on the main axis — only points outside the inset zone."""
    label_cfg = {
        # Encoders
        "DeBERTa-base":   (0.000, 0.9012, -6,  -12, "right"),
        "DeBERTa-large":  (0.000, 0.9012, -6,    8, "right"),
        # Llama (all $0, low y)
        "Llama P1":       (0.000, 0.746,   8,    0, "left"),
        "Llama P2":       (0.000, 0.818,   8,    0, "left"),
        "Llama P3":       (0.000, 0.779,   8,  -10, "left"),
        "Llama P4":       (0.000, 0.789,   8,   10, "left"),
        # GPT-4o
        "GPT-4o P1":      (0.204, 0.840,   7,    8, "left"),
        "GPT-4o P2":      (0.272, 0.829,   7,  -11, "left"),
        "GPT-4o P3":      (0.375, 0.848,   7,    8, "left"),
        "GPT-4o P4":      (0.410, 0.855,   7,  -11, "left"),
        "GPT-4o P5":      (3.520, 0.840,   7,    8, "left"),
        # Claude
        "Claude P1":      (0.310, 0.874,   7,    8, "left"),
        "Claude P2":      (0.399, 0.884,   7,  -11, "left"),  # FIX 2: cost corrected
        "Claude P3":      (2.235, 0.885,   7,    8, "left"),
        "Claude P4":      (2.800, 0.805,   7,    8, "left"),
        # GPT-5
        "GPT-5 P1":       (14.83, 0.759,   7,    8, "left"),
        "GPT-5 P3":       (17.54, 0.841,   7,  -11, "left"),
        "GPT-5 P4":       (17.54, 0.869,   7,    8, "left"),
        # v5 (outside inset)
        "v5 Ensemble":    (0.288, 0.895,   7,    8, "left"),
    }
    for name, (x, y, ox, oy, ha) in label_cfg.items():
        ax.annotate(name, xy=(x, y), xytext=(ox, oy),
                    textcoords="offset points", fontsize=8.5,
                    ha=ha, va="center", color="#222")

def add_labels_inset(axins):
    """Dense labels inside the inset — hybrid cluster + best encoders."""
    cfg = {
        "DeBERTa-base":  (0.000, 0.9012,  5,  -11, "left"),
        "DeBERTa-large": (0.000, 0.9012,  5,    8, "left"),
        "v4 0.85":       (0.006, 0.9050, -5,  -11, "right"),
        "v4 0.90 BEST":  (0.007, 0.9062, -5,    8, "right"),
        "v4 0.95":       (0.013, 0.9062,  5,    8, "left"),
        "v1 0.85":       (0.009, 0.904,   5,    8, "left"),
        "v1 0.90":       (0.013, 0.9012,  5,  -11, "left"),  # FIX 2: cost corrected
        "v1 0.95":       (0.023, 0.898,   5,    8, "left"),  # FIX 2: cost corrected
        "v2 0.85":       (0.066, 0.902,   5,    8, "left"),
        "v2 0.90":       (0.074, 0.9012,  5,  -11, "left"),
        "v2 0.95":       (0.126, 0.896,   5,    8, "left"),
        "v3 0.85":       (0.137, 0.900,   5,    8, "left"),
        "v3 0.90":       (0.152, 0.899,   5,  -11, "left"),
        "v3 0.95":       (0.258, 0.894,   5,    8, "left"),
        "v5 Ensemble":   (0.288, 0.895,   5,  -11, "left"),
    }
    for name, (x, y, ox, oy, ha) in cfg.items():
        axins.annotate(name, xy=(x, y), xytext=(ox, oy),
                       textcoords="offset points", fontsize=7.5,
                       ha=ha, va="center", color="#111")

# ─── MAIN FIGURE ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(15, 8.5))

# Plot all points on main
plot_points(ax, POINTS, show_labels=True)

# Pareto frontier
ax.plot([0.000, 0.007], [0.9012, 0.9062],
        color="#F28E2B", lw=1.5, ls="--", alpha=0.65, label="Pareto frontier")

# Reference lines
ax.axhline(0.333, color="gray", ls=":", lw=0.8, alpha=0.4)
ax.text(15, 0.337, "Random (33.3%)", fontsize=7.5, color="gray")
ax.axhline(0.9062, color="#F28E2B", ls=":", lw=0.8, alpha=0.35)
ax.text(15, 0.908, "Best: 90.62% (v4)", fontsize=7.5, color="#F28E2B")

# Main axis labels
add_labels_main(ax)

ax.set_xscale("symlog", linthresh=0.005)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: "$0" if x == 0 else (
        f"${x:.3f}" if 0 < x < 0.05 else (
            f"${x:.2f}" if x < 1 else f"${x:.0f}"
        )
    )
))
ax.set_xticks([0, 0.007, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 15, 20])
ax.set_ylim(0.725, 0.928)
ax.set_xlim(-0.003, 22)
ax.set_xlabel("Cost per 1,000 queries (USD)", fontsize=11)
ax.set_ylabel("Accuracy — Matched Test Set (800 samples)", fontsize=11)
ax.set_title("Cost-Accuracy Pareto Frontier  (All 10 system families · MultiNLI Matched)",
             fontsize=13, pad=12)
ax.legend(title="System Family", fontsize=8.5, title_fontsize=9.5,
          loc="lower right", framealpha=0.93, edgecolor="#ccc")
ax.grid(True, which="both", ls=":", alpha=0.3)

# ─── INSET — zoomed hybrid cluster ───────────────────────────────────────────
axins = ax.inset_axes([0.03, 0.60, 0.40, 0.37])  # [left, bottom, width, height] in axes coords

plot_points(axins, POINTS, fontsize=7.5, show_labels=False, zorder_base=3)
add_labels_inset(axins)

# Pareto line in inset
axins.plot([0.000, 0.007], [0.9012, 0.9062],
           color="#F28E2B", lw=1.2, ls="--", alpha=0.65)

axins.set_xscale("symlog", linthresh=0.003)
axins.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: "$0" if x == 0 else f"${x:.3f}"
))
axins.set_xticks([0, 0.007, 0.02, 0.07, 0.15, 0.30])
axins.set_xlim(-0.001, 0.32)
axins.set_ylim(0.888, 0.912)
axins.tick_params(labelsize=7)
axins.set_title("Hybrid cluster (zoomed)", fontsize=8, pad=4)
axins.grid(True, which="both", ls=":", alpha=0.3)

# Inset border box on main
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#888", lw=0.8, alpha=0.6)

plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.09)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
