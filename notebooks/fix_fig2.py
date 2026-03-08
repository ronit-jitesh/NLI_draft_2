#!/usr/bin/env python3
"""
Fix fig2: Cost-Accuracy Pareto Frontier — label overlap in top-left cluster.

Strategy:
- Use adjustText library if available, else manual offsets per point
- Widen figure, move legend inside top-right
- Use short display labels for crowded hybrid cluster
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# ── Hardcoded data (from confirmed results) ──────────────────────────────────
# (system_label, cost_per_1k, accuracy, strategy_group)
POINTS = [
    # Encoders — $0 cost
    ("DeBERTa-base",    0.0,   0.9012, "Encoder"),
    ("DeBERTa-large",   0.0,   0.9012, "Encoder"),

    # GPT-4o prompts
    ("GPT-4o P1",       0.204, 0.840,  "GPT-4o"),
    ("GPT-4o P2",       0.272, 0.829,  "GPT-4o"),
    ("GPT-4o P3",       0.375, 0.848,  "GPT-4o"),
    ("GPT-4o P4",       0.407, 0.855,  "GPT-4o"),
    ("GPT-4o P5",       3.52,  0.840,  "GPT-4o"),

    # Claude
    ("Claude P1",       0.31,  0.874,  "Claude Sonnet"),
    ("Claude P2",       0.41,  0.884,  "Claude Sonnet"),
    ("Claude P3",       2.235, 0.885,  "Claude Sonnet"),

    # Hybrid v1 (DeBERTa-base + GPT-4o P3)
    ("v1 θ=0.85",       0.009, 0.904,  "Hybrid v1"),
    ("v1 θ=0.90 ★",     0.011, 0.9012, "Hybrid v1"),
    ("v1 θ=0.95",       0.018, 0.898,  "Hybrid v1"),

    # Hybrid v2 (DeBERTa-base + Claude P4)
    ("v2 θ=0.85",       0.066, 0.902,  "Hybrid v2"),
    ("v2 θ=0.90",       0.074, 0.9012, "Hybrid v2"),
    ("v2 θ=0.95",       0.126, 0.896,  "Hybrid v2"),

    # Hybrid v3 (DeBERTa-base + GPT-4o 32-shot)
    ("v3 θ=0.85",       0.137, 0.900,  "Hybrid v3"),
    ("v3 θ=0.90",       0.152, 0.899,  "Hybrid v3"),
    ("v3 θ=0.95",       0.258, 0.894,  "Hybrid v3"),

    # Hybrid v4 (DeBERTa-large + GPT-4o P3)  ← BEST
    ("v4 θ=0.85",       0.006, 0.9050, "Hybrid v4 ⭐"),
    ("v4 θ=0.90 ⭐",    0.007, 0.9062, "Hybrid v4 ⭐"),
    ("v4 θ=0.95",       0.013, 0.9062, "Hybrid v4 ⭐"),

    # Hybrid v5 (Ensemble)
    ("v5 Ensemble",     0.288, 0.895,  "Hybrid v5"),

    # GPT-5 (o3-mini)
    ("GPT-5 P1",        14.83, 0.759,  "GPT-5 (o3-mini)"),
    ("GPT-5 P3",        17.54, 0.800,  "GPT-5 (o3-mini)"),
]

df = pd.DataFrame(POINTS, columns=["label", "cost", "accuracy", "group"])

# ── Colour + marker per group ─────────────────────────────────────────────────
GROUP_STYLE = {
    "Encoder":         {"color": "#4878CF", "marker": "D", "zorder": 5},
    "GPT-4o":          {"color": "#6ACC65", "marker": "s", "zorder": 4},
    "Claude Sonnet":   {"color": "#D65F5F", "marker": "^", "zorder": 4},
    "Hybrid v1":       {"color": "#B47CC7", "marker": "o", "zorder": 4},
    "Hybrid v2":       {"color": "#C4AD66", "marker": "o", "zorder": 4},
    "Hybrid v3":       {"color": "#77BEDB", "marker": "o", "zorder": 4},
    "Hybrid v4 ⭐":    {"color": "#F28E2B", "marker": "*", "zorder": 6, "s": 220},
    "Hybrid v5":       {"color": "#59A14F", "marker": "P", "zorder": 4},
    "GPT-5 (o3-mini)": {"color": "#E15759", "marker": "X", "zorder": 3},
}

# ── Manual label offsets (x_pt, y_pt) ─────────────────────────────────────────
# Positive x = right, negative x = left; positive y = up, negative y = down
OFFSETS = {
    # Encoders — both at $0, stagger vertically
    "DeBERTa-base":   (6,  -12),
    "DeBERTa-large":  (6,   6),

    # Hybrid v4 cluster (near $0, ~0.905–0.906)
    "v4 θ=0.85":      (-70, -14),
    "v4 θ=0.90 ⭐":   (-70,   6),
    "v4 θ=0.95":      ( 6,  10),

    # Hybrid v1 cluster (near $0.009–0.018, ~0.898–0.904)
    "v1 θ=0.85":      ( 6,   8),
    "v1 θ=0.90 ★":   ( 6,  -12),
    "v1 θ=0.95":      ( 6,  -12),

    # Hybrid v2 cluster
    "v2 θ=0.85":      (-62, -14),
    "v2 θ=0.90":      ( 6,   6),
    "v2 θ=0.95":      ( 6,  -12),

    # Hybrid v3 cluster
    "v3 θ=0.85":      ( 6,   6),
    "v3 θ=0.90":      ( 6,  -12),
    "v3 θ=0.95":      ( 6,   6),

    # GPT-4o
    "GPT-4o P1":      ( 6,  -14),
    "GPT-4o P2":      ( 6,    6),
    "GPT-4o P3":      ( 6,    6),
    "GPT-4o P4":      ( 6,  -14),
    "GPT-4o P5":      ( 6,    6),

    # Claude
    "Claude P1":      ( 6,    6),
    "Claude P2":      ( 6,  -14),
    "Claude P3":      ( 6,    6),

    # GPT-5
    "GPT-5 P1":       ( 6,    6),
    "GPT-5 P3":       ( 6,  -14),

    # v5
    "v5 Ensemble":    ( 6,    6),
}

fig, ax = plt.subplots(figsize=(13, 7.5))

# Plot each group
plotted_groups = set()
for _, row in df.iterrows():
    g = row["group"]
    style = GROUP_STYLE[g]
    s = style.get("s", 100)
    label_kwarg = g if g not in plotted_groups else "_nolegend_"
    ax.scatter(
        row["cost"], row["accuracy"],
        color=style["color"], marker=style["marker"],
        s=s, zorder=style["zorder"],
        label=label_kwarg, edgecolors="white", linewidths=0.4
    )
    plotted_groups.add(g)

# Annotate with per-point offsets
for _, row in df.iterrows():
    ox, oy = OFFSETS.get(row["label"], (6, 6))
    ax.annotate(
        row["label"],
        xy=(row["cost"], row["accuracy"]),
        xytext=(ox, oy),
        textcoords="offset points",
        fontsize=8.2,
        ha="left" if ox >= 0 else "right",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5)
        if abs(ox) > 20 else None,
    )

# Pareto frontier line (just top-3)
pareto = [(0.0, 0.9012), (0.007, 0.9062)]
px, py = zip(*pareto)
ax.plot(px, py, color="#F28E2B", linewidth=1.2, linestyle="--",
        alpha=0.6, zorder=2, label="Pareto frontier")

ax.set_xscale("symlog", linthresh=0.01)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x:.2f}" if 0 < x < 1 else (f"${int(x)}" if x >= 1 else "$0")
))
ax.set_xticks([0, 0.01, 0.1, 0.5, 1, 5, 20])

ax.set_ylim(0.74, 0.915)
ax.set_xlabel("Cost per 1,000 queries (USD)", fontsize=11)
ax.set_ylabel("Accuracy — Matched Test Set (800)", fontsize=11)
ax.set_title("Cost-Accuracy Pareto Frontier", fontsize=13, pad=12)

ax.legend(title="System", fontsize=8.5, title_fontsize=9,
          loc="lower right", framealpha=0.9)

ax.grid(True, which="both", linestyle=":", alpha=0.35)
plt.tight_layout()

out = os.path.join(FIGURES_DIR, "fig2_cost_accuracy_frontier.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
