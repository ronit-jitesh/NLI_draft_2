#!/usr/bin/env python3
"""
Notebook 09 — Genre-wise and Label-wise Analysis
==================================================
Produces:
  1. Per-class classification reports (Precision / Recall / F1) for every model
  2. Genre-wise accuracy breakdown (matched test set, 5 genres)
  3. Label confusion heatmaps per model
  4. Genre × Label error matrix

Outputs:
  results/classification_reports.csv   — P/R/F1 per class per model
  results/genre_label_breakdown.csv    — genre × label accuracy
  figures/fig13_classification_report_heatmap.png
  figures/fig14_genre_label_matrix.png
  figures/fig15_per_class_bar_all_models.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = ["entailment", "neutral", "contradiction"]

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.1)


def safe_load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [SKIP] {filename} not found")
    return pd.DataFrame()


def get_classification_report(y_true, y_pred, model_name):
    """Returns a flat dict of P/R/F1 per class for a model."""
    rows = []
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    for i, label in enumerate(LABELS):
        rows.append({
            "model": model_name,
            "class": label,
            "precision": round(p[i], 4),
            "recall":    round(r[i], 4),
            "f1":        round(f[i], 4),
            "support":   int(s[i]),
        })
    return rows


# ============================================================
# Collect predictions from all result files
# ============================================================
all_reports = []

# --- Encoders ---
df_enc = safe_load("encoder_predictions_matched.csv")
if not df_enc.empty:
    for col, name in [
        ("bert_base_pred",       "BERT-base"),
        ("deberta_v3_small_pred","DeBERTa-v3-small"),
        ("roberta_base_pred",    "RoBERTa-base"),
        ("deberta_v3_base_pred", "DeBERTa-v3-base"),
        ("deberta_v3_large_pred","DeBERTa-v3-large"),
    ]:
        if col in df_enc.columns:
            all_reports += get_classification_report(
                df_enc["label_text"], df_enc[col], name
            )
    print("Encoders: classification reports computed")

# --- GPT-4o ---
df_gpt4o = safe_load("api_results_gpt4o.csv")
if not df_gpt4o.empty:
    for prompt in df_gpt4o["prompt"].unique():
        sub = df_gpt4o[df_gpt4o["prompt"] == prompt]
        all_reports += get_classification_report(
            sub["label_true"], sub["predicted_label"], f"GPT-4o {prompt.split('_')[0]}"
        )
    print("GPT-4o: classification reports computed")

# --- Claude ---
df_claude = safe_load("api_results_claude.csv")
if not df_claude.empty:
    for prompt in df_claude["prompt"].unique():
        sub = df_claude[df_claude["prompt"] == prompt]
        valid = sub[sub["predicted_label"] != "unknown"]
        if len(valid) > 50:
            all_reports += get_classification_report(
                valid["label_true"], valid["predicted_label"],
                f"Claude {prompt.split('_')[0]}"
            )
    print("Claude: classification reports computed")

# --- Hybrid v4 (best matched) ---
df_v4 = safe_load("hybrid_v4_results.csv")
if not df_v4.empty:
    sub = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
    if not sub.empty:
        all_reports += get_classification_report(
            sub["label_true"], sub["label_pred"], "Hybrid v4 θ=0.90"
        )
    print("Hybrid v4: classification report computed")

# --- Hybrid v1 (best mismatched) ---
df_v1 = safe_load("hybrid_v1_results.csv")
if not df_v1.empty:
    sub = df_v1[(df_v1["set"] == "matched") & (df_v1["threshold"] == 0.90)]
    if not sub.empty:
        all_reports += get_classification_report(
            sub["label_true"], sub["label_pred"], "Hybrid v1 θ=0.90"
        )
    print("Hybrid v1: classification report computed")

# Save
df_reports = pd.DataFrame(all_reports)
if not df_reports.empty:
    df_reports.to_csv(os.path.join(RESULTS_DIR, "classification_reports.csv"), index=False)
    print(f"\nSaved: results/classification_reports.csv ({len(df_reports)} rows)")

    # Print readable summary
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT SUMMARY")
    print("="*70)
    for model in df_reports["model"].unique():
        sub = df_reports[df_reports["model"] == model]
        print(f"\n{model}")
        print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*56}")
        for _, row in sub.iterrows():
            print(f"  {row['class']:<16} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f} {row['support']:>10}")


# ============================================================
# Figure 13: Classification Report Heatmap (F1 per class)
# ============================================================
def plot_classification_heatmap(df_reports):
    if df_reports.empty:
        return

    # Pivot: rows = models, columns = classes, values = F1
    pivot = df_reports.pivot_table(
        index="model", columns="class", values="f1"
    ).reindex(columns=LABELS)

    # Sort by mean F1
    pivot["mean_f1"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_f1", ascending=False).drop("mean_f1", axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, len(pivot) * 0.5 + 2)),
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    metrics = ["precision", "recall", "f1"]
    titles  = ["Precision per Class", "Recall per Class", "F1-Score per Class"]
    cmaps   = ["Blues", "Greens", "Oranges"]

    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        piv = df_reports.pivot_table(
            index="model", columns="class", values=metric
        ).reindex(columns=LABELS).reindex(pivot.index)

        sns.heatmap(
            piv, ax=ax, annot=True, fmt=".3f",
            cmap=cmap, vmin=0.70, vmax=1.0,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8}
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("" if ax != axes[0] else "Model")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=9)

    plt.suptitle("Per-Class Precision / Recall / F1 — All Models (Matched 800)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig13_classification_report_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Genre-wise analysis
# ============================================================
def compute_genre_breakdown():
    """Compute per-genre accuracy for encoders + hybrid systems."""
    df_enc = safe_load("encoder_predictions_matched.csv")
    if df_enc.empty or "genre" not in df_enc.columns:
        print("  [SKIP] genre column not in encoder_predictions_matched.csv")
        print("  Add genre column in 01_data_preparation.py (join on MultiNLI genre field)")
        return pd.DataFrame()

    genres  = df_enc["genre"].unique()
    results = []

    models = {
        "DeBERTa-v3-base":  "deberta_v3_base_pred",
        "DeBERTa-v3-large": "deberta_v3_large_pred",
        "RoBERTa-base":     "roberta_base_pred",
        "BERT-base":        "bert_base_pred",
    }

    for genre in sorted(genres):
        g = df_enc[df_enc["genre"] == genre]
        for model_name, col in models.items():
            if col not in g.columns:
                continue
            acc = (g["label_text"] == g[col]).mean()
            # per-class breakdown
            for label in LABELS:
                subset = g[g["label_text"] == label]
                if len(subset) == 0:
                    continue
                label_acc = (subset["label_text"] == subset[col]).mean()
                results.append({
                    "genre": genre, "model": model_name,
                    "class": label, "accuracy": round(label_acc, 4),
                    "n": len(subset), "overall_genre_acc": round(acc, 4)
                })

    # Hybrid v4 if genre column present in hybrid results
    df_v4 = safe_load("hybrid_v4_results.csv")
    if not df_v4.empty and "genre" in df_v4.columns:
        sub = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
        for genre in sorted(sub["genre"].unique()):
            g = sub[sub["genre"] == genre]
            acc = (g["label_true"] == g["label_pred"]).mean()
            for label in LABELS:
                subset = g[g["label_true"] == label]
                if len(subset) == 0:
                    continue
                label_acc = (subset["label_true"] == subset["label_pred"]).mean()
                results.append({
                    "genre": genre, "model": "Hybrid v4",
                    "class": label, "accuracy": round(label_acc, 4),
                    "n": len(subset), "overall_genre_acc": round(acc, 4)
                })

    df_genre = pd.DataFrame(results)
    if not df_genre.empty:
        df_genre.to_csv(
            os.path.join(RESULTS_DIR, "genre_label_breakdown.csv"), index=False
        )
        print(f"Saved: results/genre_label_breakdown.csv ({len(df_genre)} rows)")
    return df_genre


# ============================================================
# Figure 14: Genre × Label accuracy matrix (DeBERTa-v3-base)
# ============================================================
def plot_genre_label_matrix(df_genre):
    if df_genre.empty:
        return

    # Focus on DeBERTa-v3-base for primary plot
    sub = df_genre[df_genre["model"] == "DeBERTa-v3-base"]
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="genre", columns="class", values="accuracy"
    ).reindex(columns=LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap="YlOrRd_r", vmin=0.82, vmax=1.0,
        linewidths=0.5, cbar_kws={"label": "Accuracy"}
    )
    ax.set_title("Genre × Label Accuracy — DeBERTa-v3-base (Matched 800)", pad=12)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Genre")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig14_genre_label_matrix.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Figure 15: Per-class F1 bar — all key models side by side
# ============================================================
def plot_per_class_bar_all_models(df_reports):
    if df_reports.empty:
        return

    # Select key models for readability
    key_models = [
        "BERT-base", "DeBERTa-v3-base",
        "GPT-4o P1", "GPT-4o P4",
        "Claude P1", "Claude P3",
        "Hybrid v4 θ=0.90"
    ]
    df_plot = df_reports[df_reports["model"].isin(key_models)].copy()
    if df_plot.empty:
        return

    # Colour per class
    palette = {
        "entailment":    "#4ade80",
        "neutral":       "#38bdf8",
        "contradiction": "#f472b6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, label in zip(axes, LABELS):
        sub = df_plot[df_plot["class"] == label]
        sub = sub.set_index("model").reindex(key_models).dropna()
        bars = ax.barh(
            sub.index, sub["f1"],
            color=palette[label], alpha=0.85, edgecolor="white", linewidth=0.4
        )
        ax.set_xlim(0.70, 1.0)
        ax.set_title(f"{label.capitalize()} F1", fontsize=11)
        ax.set_xlabel("F1 Score")
        ax.axvline(0.90, color="#555", linestyle="--", linewidth=0.7, alpha=0.5)

        for bar, val in zip(bars, sub["f1"]):
            ax.text(
                val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8
            )

    axes[0].set_ylabel("Model")
    plt.suptitle("Per-Class F1 Score — Key Models (Matched 800 samples)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig15_per_class_bar_all_models.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Run all
# ============================================================
if not df_reports.empty:
    plot_classification_heatmap(df_reports)
    plot_per_class_bar_all_models(df_reports)

df_genre = compute_genre_breakdown()
plot_genre_label_matrix(df_genre)

print("\nNotebook 09 complete.")
print("Outputs: results/classification_reports.csv, results/genre_label_breakdown.csv")
print("Figures: fig13_classification_report_heatmap.png, fig14_genre_label_matrix.png, fig15_per_class_bar_all_models.png")
