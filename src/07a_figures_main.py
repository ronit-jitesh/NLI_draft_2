#!/usr/bin/env python3
"""
Notebook 07 — Figures
=======================
Generates all 12 publication-quality figures for the report.

Figures:
    1.  Strategy Accuracy Bar (matched)
    2.  Cost-Accuracy Frontier  ← uses fix_fig2.py for the full version
    3.  Matched vs Mismatched Comparison  (DeBERTa, GPT-4o P4, Hybrid v1, v2, v4)
    4.  Per-Class F1 Grouped Bar  (DeBERTa, Hybrid v2, Hybrid v4)
    5-8. Confusion Matrices (DeBERTa, GPT-4o P4, Claude P3, Hybrid v2)
    8b. Hybrid v3 confusion matrix
    9.  Genre Heatmap (accuracy per genre per system)
    10. Hybrid v2 Threshold Trade-off  (title now correctly says "Hybrid v2")
    11. Ensemble Gate Breakdown (v5)
    12. Gating Strategy Comparison (v4 vs v5)

BUG FIXES applied vs original:
    FIX 1 — Fig 1: Removed duplicate Claude bars.
             CSV loop produced "Claude Sonnet P1/P3" keys; verified_points produced
             "Claude Sonnet 4.5 P*" keys -> both showed up. Fixed by deleting the CSV
             loop for Claude and relying entirely on the verified_points dict, which
             covers all four prompts with the canonical naming.
    FIX 2 — Fig 2 (fix_fig2.py data): Hybrid v1 theta=0.90 cost hardcoded as $0.011 but
             cost_summary.csv and §5.2 report $0.013.  Corrected to 0.013.
             Also corrected Claude P2 cost from 0.410 -> 0.399 (from CSV).
    FIX 3 — Fig 10: Title changed from generic "Hybrid Architecture Trade-off" to
             "Hybrid v2 Architecture Trade-off" since the data source is
             hybrid_v2_results.csv.
    FIX 4 — Fig 3: Added Hybrid v1 (best mismatched 91.3%) and Hybrid v4 to the
             Matched vs Mismatched comparison figure.
    FIX 5 — Fig 4: Added Hybrid v4 (theta=0.90) per-class F1 bars alongside DeBERTa
             and Hybrid v2.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

# Labels order
LABELS = ["entailment", "neutral", "contradiction"]


# ============================================================
# Helper Functions
# ============================================================
def safe_load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    f1_cls = f1_score(y_true, y_pred, average=None, labels=LABELS)
    return acc, f1, f1_cls


# ============================================================
# Figure 1: Strategy Accuracy Bar (matched)
# FIX 1: Removed duplicate Claude bars by eliminating the CSV
#         loop for Claude and using only verified_points dict.
# ============================================================
def plot_strategy_accuracy_bar(df_cost):
    if df_cost.empty:
        return
    print("Generating Fig 1: Strategy Accuracy Bar...")

    accuracies = {"Random Baseline": 0.333}
    colors = {"Random Baseline": "darkgrey"}

    # --- Encoders (from CSV) ---
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
            acc = accuracy_score(df_enc_m["label_text"], df_enc_m[f"{model}_pred"])
            name = model.replace("_", "-").title().replace("V3", "v3")
            if "Deberta" in name:
                name = name.replace("Deberta", "DeBERTa")
            accuracies[name] = acc
            colors[name] = "steelblue"

    # --- GPT-4o (from CSV) ---
    df_gpt4o = safe_load("api_results_gpt4o.csv")
    if not df_gpt4o.empty:
        for prompt in df_gpt4o["prompt"].unique():
            sub = df_gpt4o[df_gpt4o["prompt"] == prompt]
            acc = accuracy_score(sub["label_true"], sub["predicted_label"])
            name = f"GPT-4o {prompt.split('_')[0]}"
            accuracies[name] = acc
            colors[name] = "mediumseagreen"

    # --- Llama (from CSV) ---
    df_llama = safe_load("api_results_llama.csv")
    if not df_llama.empty:
        for prompt in df_llama["prompt"].unique():
            sub = df_llama[df_llama["prompt"] == prompt]
            acc = accuracy_score(sub["label_true"], sub["predicted_label"])
            name = f"Llama 3.3 {prompt.split('_')[0]}"
            accuracies[name] = acc
            colors[name] = "coral"

    # --- GPT-5 (from CSV) ---
    df_gpt5 = safe_load("api_results_gpt5.csv")
    if not df_gpt5.empty:
        for prompt in df_gpt5["prompt"].unique():
            sub = df_gpt5[df_gpt5["prompt"] == prompt]
            acc = accuracy_score(sub["label_true"], sub["predicted_label"])
            name = f"GPT-5 {prompt.split('_')[0]}"
            accuracies[name] = acc
            colors[name] = "gold"

    # ----------------------------------------------------------------
    # FIX 1: Claude bars -- use ONLY verified_points (no CSV loop).
    # The CSV loop used "Claude Sonnet P1/P3" naming while
    # verified_points used "Claude Sonnet 4.5 P*" -> duplicates appeared.
    # Canonical name is "Claude Sonnet 4.5 P*" for all four prompts.
    # ----------------------------------------------------------------
    verified_points = {
        "Claude Sonnet 4.5 P1": 0.874,
        "Claude Sonnet 4.5 P2": 0.884,
        "Claude Sonnet 4.5 P3": 0.885,
        "Claude Sonnet 4.5 P4": 0.805,
        # Llama 3.3 70B -- overrides (CSV-derived)
        "Llama 3.3 P1": 0.746,
        "Llama 3.3 P2": 0.818,
        "Llama 3.3 P3": 0.779,
        "Llama 3.3 P4": 0.789,
        # GPT-4o -- overrides (confirmed numbers)
        "GPT-4o P1": 0.840,
        "GPT-4o P2": 0.829,
        "GPT-4o P3": 0.848,
        "GPT-4o P4": 0.855,
        # GPT-5
        "GPT-5 P3": 0.841,
        "GPT-5 P4": 0.869,
    }
    for name, acc in verified_points.items():
        accuracies[name] = acc
        if "Llama" in name:
            colors[name] = "coral"
        elif "Claude" in name:
            colors[name] = "lightseagreen"
        elif "GPT-4o" in name:
            colors[name] = "mediumseagreen"
        elif "GPT-5" in name:
            colors[name] = "gold"

    # --- Hybrid systems (computed live from CSVs) ---
    for label, fname, color in [
        ("Hybrid v2 (theta=0.90)", "hybrid_v2_results.csv", "crimson"),
        ("Hybrid v3 (theta=0.90)", "hybrid_v3_results.csv", "darkviolet"),
        ("Hybrid v4 (theta=0.90)", "hybrid_v4_results.csv", "orange"),
    ]:
        df_h = safe_load(fname)
        if not df_h.empty:
            sub = df_h[(df_h["set"] == "matched") & (df_h["threshold"] == 0.90)]
            if not sub.empty:
                accuracies[label] = accuracy_score(sub["label_true"], sub["label_pred"])
                colors[label] = color

    df_v5 = safe_load("hybrid_v5_results.csv")
    if not df_v5.empty:
        sub = df_v5[df_v5["set"] == "matched"]
        if not sub.empty:
            accuracies["Hybrid v5 (Ensemble)"] = accuracy_score(sub["label_true"], sub["label_pred"])
            colors["Hybrid v5 (Ensemble)"] = "navy"

    df_v5b = safe_load("hybrid_v5b_results.csv")
    if not df_v5b.empty:
        sub = df_v5b[df_v5b["set"] == "matched"]
        if not sub.empty:
            accuracies["Hybrid v5b (Tiered)"] = accuracy_score(sub["label_true"], sub["label_pred"])
            colors["Hybrid v5b (Tiered)"] = "slateblue"

    df_v5c = safe_load("hybrid_v5c_results.csv")
    if not df_v5c.empty:
        sub = df_v5c[df_v5c["set"] == "matched"]
        if not sub.empty:
            accuracies["Hybrid v5c (Claude)"] = accuracy_score(sub["label_true"], sub["label_pred"])
            colors["Hybrid v5c (Claude)"] = "teal"

    # --- Plot ---
    df_plot = pd.DataFrame(list(accuracies.items()), columns=["System", "Accuracy"])
    df_plot = df_plot.sort_values("Accuracy", ascending=True)

    plt.figure(figsize=(12, max(6, len(df_plot) * 0.35)))
    ax = sns.barplot(
        x="Accuracy", y="System", data=df_plot,
        palette=[colors[s] for s in df_plot["System"]]
    )

    ax.set_xlim(0, 1.0)
    ax.set_title("NLI Accuracy on Matched Test Set (800 samples)", pad=15)
    ax.set_xlabel("Accuracy")

    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.01, p.get_y() + p.get_height() / 2,
            f"{width * 100:.1f}%", ha="left", va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig1_strategy_accuracy_bar.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: fig1_strategy_accuracy_bar.png")


# ============================================================
# Figure 2: Cost-Accuracy Frontier (simple fallback version)
# FIX 2 is applied in fix_fig2.py (corrected v1 theta=0.90 cost
# from $0.011 to $0.013, and Claude P2 from $0.410 to $0.399).
# ============================================================
def plot_cost_accuracy_frontier(df_cost):
    if df_cost.empty:
        return
    print("Generating Fig 2 (fallback): Cost-Accuracy Frontier...")

    accuracies = {}

    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
            nm = model.replace("_", "-").title()
            if "Deberta" in nm:
                nm = nm.replace("Deberta", "DeBERTa")
            accuracies[nm] = accuracy_score(df_enc_m["label_text"], df_enc_m[f"{model}_pred"])

    df_gpt4o = safe_load("api_results_gpt4o.csv")
    if not df_gpt4o.empty:
        for p in df_gpt4o["prompt"].unique():
            sub = df_gpt4o[df_gpt4o["prompt"] == p]
            accuracies[f"GPT-4o {p}"] = accuracy_score(sub["label_true"], sub["predicted_label"])

    df_claude = safe_load("api_results_claude.csv")
    if not df_claude.empty:
        for p in df_claude["prompt"].unique():
            sub = df_claude[df_claude["prompt"] == p]
            valid = sub[sub["predicted_label"] != "unknown"]
            if len(valid) == 0:
                continue
            accuracies[f"Claude Sonnet {p}"] = accuracy_score(
                valid["label_true"], valid["predicted_label"]
            )

    for version, fname in [
        ("v2", "hybrid_v2_results.csv"),
        ("v3", "hybrid_v3_results.csv"),
        ("v4", "hybrid_v4_results.csv"),
    ]:
        df_h = safe_load(fname)
        if not df_h.empty:
            for th in [0.85, 0.90, 0.95]:
                sub = df_h[(df_h["set"] == "matched") & (df_h["threshold"] == th)]
                if len(sub) > 0:
                    s = " (BEST)" if (version == "v4" and th == 0.90) else ""
                    accuracies[f"Hybrid {version} t={th}{s}"] = accuracy_score(
                        sub["label_true"], sub["label_pred"]
                    )

    df_v5 = safe_load("hybrid_v5_results.csv")
    if not df_v5.empty:
        sub = df_v5[df_v5["set"] == "matched"]
        if not sub.empty:
            accuracies["Hybrid v5 (Ensemble Gate)"] = accuracy_score(
                sub["label_true"], sub["label_pred"]
            )

    df_cost = df_cost.copy()
    df_cost["Accuracy"] = df_cost["system"].map(accuracies)
    df_plot = df_cost.dropna(subset=["Accuracy"])

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="cost_per_1k", y="Accuracy",
        hue="strategy", style="strategy",
        s=150, data=df_plot, palette="tab10"
    )

    for _, row in df_plot.iterrows():
        sys_label = row["system"]
        if "P" in sys_label:
            sys_label = sys_label.split("_")[0]
        offset_y = 0.005 if "Hybrid" in sys_label else -0.015
        if "P1" in sys_label:
            offset_y = 0.015
        plt.annotate(sys_label, (row["cost_per_1k"], row["Accuracy"]),
                     xytext=(5, offset_y * 1000),
                     textcoords="offset points", fontsize=8)

    plt.title("Cost-Accuracy Pareto Frontier", pad=15)
    plt.xlabel("Cost per 1,000 queries (USD)")
    plt.ylabel("Accuracy on Matched Test Set")
    plt.xscale("symlog", linthresh=0.1)
    ticks = [0, 0.1, 1, 10, 100]
    plt.xticks(ticks, [f"${t}" for t in ticks])
    plt.legend(title="Strategy / Architecture", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig2_cost_accuracy_frontier.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: fig2_cost_accuracy_frontier.png")
    print("  NOTE: Run fix_fig2.py for the full publication version with inset zoom panel.")


# ============================================================
# Figure 3: Matched vs Mismatched Comparison
# FIX 4: Added Hybrid v1 (best mismatched 91.3%) and Hybrid v4
# ============================================================
def plot_matched_vs_mismatched():
    print("Generating Fig 3: Matched vs Mismatched Comparison...")

    data = []

    # DeBERTa-v3-base
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_enc_mm = safe_load("encoder_predictions_mm.csv")
    if not df_enc_m.empty and not df_enc_mm.empty:
        c1 = accuracy_score(df_enc_m["label_text"], df_enc_m["deberta_v3_base_pred"])
        c2 = accuracy_score(df_enc_mm["label_text"], df_enc_mm["deberta_v3_base_pred"])
        data += [
            {"System": "DeBERTa-v3-base", "Set": "Matched",    "Accuracy": c1},
            {"System": "DeBERTa-v3-base", "Set": "Mismatched", "Accuracy": c2},
        ]

    # GPT-4o P4
    df_gpt4o    = safe_load("api_results_gpt4o.csv")
    df_gpt4o_mm = safe_load("api_results_gpt4o_mm.csv")
    if not df_gpt4o.empty and not df_gpt4o_mm.empty:
        m  = df_gpt4o[df_gpt4o["prompt"] == "P4_few_shot_cot"]
        mm = df_gpt4o_mm[df_gpt4o_mm["prompt"] == "P4_few_shot_cot"]
        if len(m) > 0 and len(mm) > 0:
            data += [
                {"System": "GPT-4o P4 (CoT)", "Set": "Matched",
                 "Accuracy": accuracy_score(m["label_true"],  m["predicted_label"])},
                {"System": "GPT-4o P4 (CoT)", "Set": "Mismatched",
                 "Accuracy": accuracy_score(mm["label_true"], mm["predicted_label"])},
            ]

    # FIX 4 -- Hybrid v1 (best mismatched system, 91.3%)
    df_v1 = safe_load("hybrid_v1_results.csv")
    if not df_v1.empty:
        m  = df_v1[(df_v1["set"] == "matched")    & (df_v1["threshold"] == 0.90)]
        mm = df_v1[(df_v1["set"] == "mismatched") & (df_v1["threshold"] == 0.90)]
        if len(m) > 0 and len(mm) > 0:
            data += [
                {"System": "Hybrid v1 (best MM)",
                 "Set": "Matched",    "Accuracy": accuracy_score(m["label_true"],  m["label_pred"])},
                {"System": "Hybrid v1 (best MM)",
                 "Set": "Mismatched", "Accuracy": accuracy_score(mm["label_true"], mm["label_pred"])},
            ]

    # Hybrid v2
    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        m  = df_v2[(df_v2["set"] == "matched")    & (df_v2["threshold"] == 0.90)]
        mm = df_v2[(df_v2["set"] == "mismatched") & (df_v2["threshold"] == 0.90)]
        if len(m) > 0 and len(mm) > 0:
            data += [
                {"System": "Hybrid v2", "Set": "Matched",
                 "Accuracy": accuracy_score(m["label_true"],  m["label_pred"])},
                {"System": "Hybrid v2", "Set": "Mismatched",
                 "Accuracy": accuracy_score(mm["label_true"], mm["label_pred"])},
            ]

    # FIX 4 -- Hybrid v4 (best matched system, 90.62%)
    df_v4 = safe_load("hybrid_v4_results.csv")
    if not df_v4.empty:
        m  = df_v4[(df_v4["set"] == "matched")    & (df_v4["threshold"] == 0.90)]
        mm = df_v4[(df_v4["set"] == "mismatched") & (df_v4["threshold"] == 0.90)]
        if len(m) > 0 and len(mm) > 0:
            data += [
                {"System": "Hybrid v4 (best M)",
                 "Set": "Matched",    "Accuracy": accuracy_score(m["label_true"],  m["label_pred"])},
                {"System": "Hybrid v4 (best M)",
                 "Set": "Mismatched", "Accuracy": accuracy_score(mm["label_true"], mm["label_pred"])},
            ]

    if not data:
        print("  Skipping Fig 3 -- no data found")
        return

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(13, 6))
    ax = sns.barplot(
        x="System", y="Accuracy", hue="Set", data=df_plot,
        palette=["steelblue", "darkorange"]
    )
    ax.set_ylim(0.80, 0.95)
    ax.set_title("Cross-Genre Generalisation: Matched vs Mismatched (theta=0.90)", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2, height + 0.001,
                f"{height * 100:.1f}%", ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig3_matched_vs_mismatched.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: fig3_matched_vs_mismatched.png")


# ============================================================
# Figure 4: Per-Class F1 Grouped Bar
# FIX 5: Added Hybrid v4 (theta=0.90) alongside DeBERTa and v2
# ============================================================
def plot_per_class_f1():
    print("Generating Fig 4: Per-Class F1 Grouped Bar...")

    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if df_enc_m.empty:
        return

    f1_deberta = f1_score(
        df_enc_m["label_text"], df_enc_m["deberta_v3_base_pred"],
        average=None, labels=LABELS
    )

    # Hybrid v2
    df_v2 = safe_load("hybrid_v2_results.csv")
    f1_v2 = None
    if not df_v2.empty:
        sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        if not sub.empty:
            f1_v2 = f1_score(sub["label_true"], sub["label_pred"], average=None, labels=LABELS)

    # FIX 5: Hybrid v4
    df_v4 = safe_load("hybrid_v4_results.csv")
    f1_v4 = None
    if not df_v4.empty:
        sub = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
        if not sub.empty:
            f1_v4 = f1_score(sub["label_true"], sub["label_pred"], average=None, labels=LABELS)

    n_sys = 1 + (1 if f1_v2 is not None else 0) + (1 if f1_v4 is not None else 0)
    x = np.arange(len(LABELS))
    width = 0.25 if n_sys == 3 else 0.35
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x + offsets[0], f1_deberta, width, label="DeBERTa-v3-base", color="steelblue")

    idx = 1
    if f1_v2 is not None:
        ax.bar(x + offsets[idx], f1_v2, width, label="Hybrid v2 (theta=0.90)", color="crimson")
        idx += 1
    if f1_v4 is not None:
        ax.bar(x + offsets[idx], f1_v4, width, label="Hybrid v4 (theta=0.90) [BEST]", color="orange")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison (Matched Test Set)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.title() for lbl in LABELS])
    ax.legend()
    ax.set_ylim(0.70, 1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_per_class_f1.png"), dpi=300)
    plt.close()
    print("  Saved: fig4_per_class_f1.png")


# ============================================================
# Figures 5-8: Confusion Matrices
# ============================================================
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        vmin=0, vmax=1
    )
    plt.title(f"Confusion Matrix: {title}", pad=15)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()


def plot_confusion_matrices():
    print("Generating Figs 5-8: Confusion Matrices...")

    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        plot_cm(
            df_enc_m["label_text"], df_enc_m["deberta_v3_base_pred"],
            "DeBERTa-v3-base (Matched)", "fig5_cm_deberta.png"
        )

    df_gpt4o = safe_load("api_results_gpt4o.csv")
    if not df_gpt4o.empty:
        m = df_gpt4o[df_gpt4o["prompt"] == "P4_few_shot_cot"]
        if len(m) > 0:
            plot_cm(m["label_true"], m["predicted_label"],
                    "GPT-4o P4 CoT (Matched)", "fig6_cm_gpt4o.png")

    df_claude = safe_load("api_results_claude.csv")
    if not df_claude.empty:
        m = df_claude[df_claude["prompt"] == "P3_few_shot"]
        valid = m[m["predicted_label"] != "unknown"]
        if len(valid) > 0:
            plot_cm(valid["label_true"], valid["predicted_label"],
                    "Claude Sonnet 4.5 P3 (Matched)", "fig7_cm_claude.png")

    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        if len(sub) > 0:
            plot_cm(sub["label_true"], sub["label_pred"],
                    "Hybrid v2 theta=0.90 (Matched)", "fig8_cm_hybrid.png")

    df_v3 = safe_load("hybrid_v3_results.csv")
    if not df_v3.empty:
        sub = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)]
        if len(sub) > 0:
            plot_cm(sub["label_true"], sub["label_pred"],
                    "Hybrid v3 theta=0.90 (Matched)", "fig8b_cm_hybrid_v3.png")

    print("  Saved: fig5 through fig8b confusion matrices")


# ============================================================
# Figure 9: Genre Heatmap
# ============================================================
def plot_genre_heatmap():
    print("Generating Fig 9: Genre Heatmap...")
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_v2 = safe_load("hybrid_v2_results.csv")

    if df_enc_m.empty or df_v2.empty:
        return

    genres = sorted(df_enc_m["genre"].dropna().unique())
    systems = ["DeBERTa", "Hybrid v2", "Hybrid v3", "Hybrid v4"]
    data = np.zeros((len(systems), len(genres)))

    sub_v2 = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]

    df_v3 = safe_load("hybrid_v3_results.csv")
    sub_v3 = (
        df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)]
        if not df_v3.empty else pd.DataFrame()
    )

    df_v4 = safe_load("hybrid_v4_results.csv")
    sub_v4 = (
        df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
        if not df_v4.empty else pd.DataFrame()
    )

    for i, g in enumerate(genres):
        g_enc = df_enc_m[df_enc_m["genre"] == g]
        if len(g_enc) > 0:
            data[0, i] = accuracy_score(g_enc["label_text"], g_enc["deberta_v3_base_pred"])

        g_v2 = sub_v2[sub_v2["genre"] == g]
        if len(g_v2) > 0:
            data[1, i] = accuracy_score(g_v2["label_true"], g_v2["label_pred"])

        if not sub_v3.empty:
            g_v3 = sub_v3[sub_v3["genre"] == g]
            if len(g_v3) > 0:
                data[2, i] = accuracy_score(g_v3["label_true"], g_v3["label_pred"])

        if not sub_v4.empty:
            g_v4 = sub_v4[sub_v4["genre"] == g]
            if len(g_v4) > 0:
                data[3, i] = accuracy_score(g_v4["label_true"], g_v4["label_pred"])

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        data, annot=True, fmt=".1%", cmap="YlGnBu",
        xticklabels=genres, yticklabels=systems, vmin=0.80
    )
    plt.title("Accuracy across Genres (Matched Set, theta=0.90)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig9_genre_heatmap.png"), dpi=300)
    plt.close()
    print("  Saved: fig9_genre_heatmap.png")


# ============================================================
# Figure 10: Hybrid v2 Threshold Trade-off
# FIX 3: Title now explicitly identifies "Hybrid v2" as the source.
# ============================================================
def plot_hybrid_threshold():
    print("Generating Fig 10: Hybrid v2 Threshold Trade-off...")
    df_v2 = safe_load("hybrid_v2_results.csv")
    if df_v2.empty:
        return

    thresholds, accs, apipct = [], [], []
    sub_m = df_v2[df_v2["set"] == "matched"]

    for th in sorted(sub_m["threshold"].unique()):
        th_data = sub_m[sub_m["threshold"] == th]
        if len(th_data) == 0:
            continue
        acc = accuracy_score(th_data["label_true"], th_data["label_pred"])
        api_pct = (th_data["source"] == "api").mean()
        thresholds.append(th)
        accs.append(acc)
        apipct.append(api_pct)

    if not thresholds:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("DeBERTa Confidence Threshold (theta)")
    ax1.set_ylabel("System Accuracy", color=color)
    ax1.plot(thresholds, accs, marker="o", color=color, linewidth=2, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Samples Routed to LLM API (%)", color=color)
    ax2.plot(thresholds, apipct, marker="s", color=color, linewidth=2,
             linestyle="--", label="% to API")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    # FIX 3: Now explicitly says "Hybrid v2" (previously said "Hybrid Architecture")
    plt.title("Hybrid v2 Architecture Trade-off (Accuracy vs API Cost)", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig10_hybrid_threshold.png"), dpi=300)
    plt.close()
    print("  Saved: fig10_hybrid_threshold.png")


# ============================================================
# Figure 11: Ensemble Gate Breakdown (Hybrid v5)
# ============================================================
def plot_ensemble_breakdown():
    print("Generating Fig 11: Ensemble Gate Breakdown...")
    df_v5 = safe_load("hybrid_v5_results.csv")
    if df_v5.empty:
        return

    sub = df_v5[df_v5["set"] == "matched"]
    ens = sub[sub["source"] == "ensemble"]
    api = sub[sub["source"] == "api"]

    if ens.empty or api.empty:
        return

    ens_acc   = accuracy_score(ens["label_true"], ens["label_pred"])
    api_acc   = accuracy_score(api["label_true"], api["label_pred"])
    total_acc = accuracy_score(sub["label_true"], sub["label_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ens_n = len(ens)
    api_n = len(api)
    total_n = len(sub)

    ax1.bar([0], [ens_n], color="steelblue",
            label=f"Unanimous ({ens_n}, {ens_n / total_n:.1%})")
    ax1.bar([0], [api_n], bottom=[ens_n], color="coral",
            label=f"Escalated ({api_n}, {api_n / total_n:.1%})")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylabel("Number of samples (800 total)")
    ax1.set_title("Gate Decision Split")
    ax1.legend(loc="upper right")
    ax1.set_xticks([])
    ax1.text(0, ens_n / 2, f"{ens_acc * 100:.1f}%\naccuracy",
             ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax1.text(0, ens_n + api_n / 2, f"{api_acc * 100:.1f}%\naccuracy",
             ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    ax2 = axes[1]
    systems    = [f"Unanimous\n({ens_n})", f"Escalated\n({api_n})", f"Total\n({total_n})"]
    accs_bar   = [ens_acc, api_acc, total_acc]
    bar_colors = ["steelblue", "coral", "slategray"]

    bars = ax2.bar(systems, accs_bar, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylim(0.3, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by Gate Decision")
    ax2.axhline(0.333, color="red",    linestyle=":", alpha=0.5,
                label="Random baseline (33.3%)")
    ax2.axhline(0.512, color="orange", linestyle="--", alpha=0.5,
                label="GPT-4o on escalated (51.0%)")
    ax2.legend(fontsize=9)

    for bar, acc in zip(bars, accs_bar):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc * 100:.1f}%", ha="center", va="bottom", fontweight="bold"
        )

    plt.suptitle("Hybrid v5: Ensemble Gate Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig11_ensemble_breakdown.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: fig11_ensemble_breakdown.png")


# ============================================================
# Figure 12: Gating Strategy Comparison (v4 vs v5)
# ============================================================
def plot_gating_comparison():
    print("Generating Fig 12: Gating Strategy Comparison...")

    df_v4 = safe_load("hybrid_v4_results.csv")
    df_v5 = safe_load("hybrid_v5_results.csv")

    if df_v4.empty or df_v5.empty:
        return

    sub_v4 = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
    sub_v5 = df_v5[df_v5["set"] == "matched"]

    if sub_v4.empty or sub_v5.empty:
        return

    v4_enc = sub_v4[sub_v4["source"] == "encoder"]
    v4_api = sub_v4[sub_v4["source"] == "api"]
    v5_ens = sub_v5[sub_v5["source"] == "ensemble"]
    v5_api = sub_v5[sub_v5["source"] == "api"]

    cats_v4 = [
        accuracy_score(v4_enc["label_true"], v4_enc["label_pred"]) if len(v4_enc) > 0 else 0,
        accuracy_score(v4_api["label_true"], v4_api["label_pred"]) if len(v4_api) > 0 else 0,
    ]
    cats_v5 = [
        accuracy_score(v5_ens["label_true"], v5_ens["label_pred"]) if len(v5_ens) > 0 else 0,
        accuracy_score(v5_api["label_true"], v5_api["label_pred"]) if len(v5_api) > 0 else 0,
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    w = 0.35

    bars1 = ax.bar(x - w / 2, cats_v4, w, color="steelblue",
                   label=f"v4: Confidence Gate  (n={len(v4_enc)}/{len(v4_api)})")
    bars2 = ax.bar(x + w / 2, cats_v5, w, color="coral",
                   label=f"v5: Ensemble Gate  (n={len(v5_ens)}/{len(v5_api)})")

    ax.set_xticks(x)
    ax.set_xticklabels(["Auto-accept / Unanimous", "Escalated / Disagreement"])
    ax.set_ylim(0.30, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Confidence Gating vs Ensemble Gating:\n"
        "Accuracy on Handled vs Escalated Samples",
        pad=12
    )
    ax.legend()
    ax.axhline(0.333, color="red", linestyle=":", alpha=0.4, label="Random baseline")

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height() * 100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig12_gating_comparison.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: fig12_gating_comparison.png")


# ============================================================
# Main
# ============================================================
def main():
    df_cost = safe_load("cost_summary.csv")

    plot_strategy_accuracy_bar(df_cost)   # Fig 1  -- FIX 1 applied
    plot_cost_accuracy_frontier(df_cost)  # Fig 2  -- FIX 2 is in fix_fig2.py
    plot_matched_vs_mismatched()          # Fig 3  -- FIX 4 applied
    plot_per_class_f1()                   # Fig 4  -- FIX 5 applied
    plot_confusion_matrices()             # Figs 5-8b
    plot_genre_heatmap()                  # Fig 9
    plot_hybrid_threshold()               # Fig 10 -- FIX 3 applied
    plot_ensemble_breakdown()             # Fig 11
    plot_gating_comparison()              # Fig 12

    print("=" * 60)
    print("FIGURES COMPLETE")
    print("=" * 60)
    print()
    print("For the publication-quality Fig 2 with inset zoom panel,")
    print("also run:  python notebooks/fix_fig2.py")


if __name__ == "__main__":
    main()
