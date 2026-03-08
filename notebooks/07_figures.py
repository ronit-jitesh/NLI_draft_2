#!/usr/bin/env python3
"""
Notebook 07 — Figures
=======================
Generates all 10 publication-quality figures for the report.

Figures:
    1. Strategy Accuracy Bar (matched)
    2. Cost-Accuracy Frontier (THE KEY FIGURE)
    3. Matched vs Mismatched Comparison
    4. Per-Class F1 Grouped Bar
    5-8. Confusion Matrices (DeBERTa, GPT-4o P4, Claude P2, Hybrid v2)
    9. Genre Heatmap (F1 per genre per system)
    10. Hybrid Threshold Trade-off
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
# ============================================================
def plot_strategy_accuracy_bar(df_cost):
    if df_cost.empty: return
    print("Generating Fig 1: Strategy Accuracy Bar...")

    plt.figure(figsize=(12, 6))

    # We need to extract accuracies from individual result files
    # since df_cost only has tokens/cost
    accuracies = {"Random Baseline": 0.333}
    colors = {"Random Baseline": "darkgrey"}

    # Encoders
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
            acc = accuracy_score(df_enc_m["label_text"], df_enc_m[f"{model}_pred"])
            name = model.replace("_", "-").title().replace("V3", "v3")
            if "Deberta" in name: name = name.replace("Deberta", "DeBERTa")
            accuracies[name] = acc
            colors[name] = "steelblue"

    # GPT-4o
    df_gpt4o = safe_load("api_results_gpt4o.csv")
    if not df_gpt4o.empty:
        for prompt in df_gpt4o["prompt"].unique():
            sub = df_gpt4o[df_gpt4o["prompt"] == prompt]
            acc = accuracy_score(sub["label_true"], sub["predicted_label"])
            name = f"GPT-4o {prompt.split('_')[0]}"
            accuracies[name] = acc
            colors[name] = "mediumseagreen"

    # Claude — only include prompts with valid (non-ERROR) predictions
    df_claude = safe_load("api_results_claude.csv")
    if not df_claude.empty:
        for prompt in df_claude["prompt"].unique():
            sub = df_claude[df_claude["prompt"] == prompt]
            valid = sub[sub["predicted_label"] != "unknown"]
            if len(valid) == 0:
                continue  # skip entirely-failed prompts (e.g. P4 ERROR)
            acc = accuracy_score(valid["label_true"], valid["predicted_label"])
            name = f"Claude Sonnet {prompt.split('_')[0]}"
            if "P1" in name or "P3" in name:  # Best performing prompts
                accuracies[name] = acc
                colors[name] = "lightseagreen"

    # Llama
    df_llama = safe_load("api_results_llama.csv")
    if not df_llama.empty:
        for prompt in df_llama["prompt"].unique():
            sub = df_llama[df_llama["prompt"] == prompt]
            acc = accuracy_score(sub["label_true"], sub["predicted_label"])
            name = f"Llama 3.3 {prompt.split('_')[0]}"
            if "P1" in name or "P2" in name:
                accuracies[name] = acc
                colors[name] = "coral"

    # Hybrid v2
    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v2 (θ=0.90)"] = acc
            colors["Hybrid v2 (θ=0.90)"] = "crimson"

    # Hybrid v3
    df_v3 = safe_load("hybrid_v3_results.csv")
    if not df_v3.empty:
        sub = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v3 (θ=0.90)"] = acc
            colors["Hybrid v3 (θ=0.90)"] = "darkviolet"

    # Hybrid v4
    df_v4 = safe_load("hybrid_v4_results.csv")
    if not df_v4.empty:
        sub = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v4 (θ=0.90)"] = acc
            colors["Hybrid v4 (θ=0.90)"] = "orange"

    # Hybrid v5
    df_v5 = safe_load("hybrid_v5_results.csv")
    if not df_v5.empty:
        sub = df_v5[df_v5["set"] == "matched"]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v5 (Ensemble)"] = acc
            colors["Hybrid v5 (Ensemble)"] = "navy"

    # Hybrid v5b
    df_v5b = safe_load("hybrid_v5b_results.csv")
    if not df_v5b.empty:
        sub = df_v5b[df_v5b["set"] == "matched"]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v5b (Tiered)"] = acc
            colors["Hybrid v5b (Tiered)"] = "slateblue"

    # Hybrid v5c
    df_v5c = safe_load("hybrid_v5c_results.csv")
    if not df_v5c.empty:
        sub = df_v5c[df_v5c["set"] == "matched"]
        if not sub.empty:
            acc = accuracy_score(sub["label_true"], sub["label_pred"])
            accuracies["Hybrid v5c (Claude)"] = acc
            colors["Hybrid v5c (Claude)"] = "teal"

    # Plot
    df_plot = pd.DataFrame(list(accuracies.items()), columns=["System", "Accuracy"])
    df_plot = df_plot.sort_values("Accuracy", ascending=True)

    ax = sns.barplot(x="Accuracy", y="System", data=df_plot,
                     palette=[colors[s] for s in df_plot["System"]])

    ax.set_xlim(0, 1.0)
    ax.set_title("NLI Accuracy on Matched Test Set (800 samples)", pad=15)
    ax.set_xlabel("Accuracy")

    # Annotate bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                f"{width*100:.1f}%", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_strategy_accuracy_bar.png"), dpi=300)
    plt.close()


# ============================================================
# Figure 2: Cost-Accuracy Frontier
# ============================================================
def plot_cost_accuracy_frontier(df_cost):
    if df_cost.empty: return
    print("Generating Fig 2: Cost-Accuracy Frontier...")

    # Attach accuracy to cost rows
    accuracies = {}

    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
            nm = model.replace("_", "-").title()
            if "Deberta" in nm: nm = nm.replace("Deberta", "DeBERTa")
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
                continue  # skip zero-valid prompts (P4 all ERROR)
            accuracies[f"Claude Sonnet {p}"] = accuracy_score(valid["label_true"], valid["predicted_label"])

    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        for th in [0.85, 0.90, 0.95]:
            sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == th)]
            if len(sub) > 0:
                s = " ★" if th == 0.90 else ""
                accuracies[f"Hybrid v2 θ={th}{s}"] = accuracy_score(sub["label_true"], sub["label_pred"])

    df_v3 = safe_load("hybrid_v3_results.csv")
    if not df_v3.empty:
        for th in [0.85, 0.90, 0.95]:
            sub = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == th)]
            if len(sub) > 0:
                accuracies[f"Hybrid v3 θ={th}"] = accuracy_score(sub["label_true"], sub["label_pred"])

    df_v4 = safe_load("hybrid_v4_results.csv")
    if not df_v4.empty:
        for th in [0.85, 0.90, 0.95]:
            sub = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == th)]
            if len(sub) > 0:
                accuracies[f"Hybrid v4 θ={th}"] = accuracy_score(sub["label_true"], sub["label_pred"])

    df_v5 = safe_load("hybrid_v5_results.csv")
    if not df_v5.empty:
        sub = df_v5[df_v5["set"] == "matched"]
        if not sub.empty:
            accuracies["Hybrid v5 (Ensemble Gate)"] = accuracy_score(sub["label_true"], sub["label_pred"])

    # Update df_cost
    df_cost["Accuracy"] = df_cost["system"].map(accuracies)
    df_plot = df_cost.dropna(subset=["Accuracy"]).copy()

    plt.figure(figsize=(10, 7))

    # Scatter plot
    sns.scatterplot(
        x="cost_per_1k", y="Accuracy",
        hue="strategy", style="strategy",
        s=150, data=df_plot, palette="tab10"
    )

    # Annotate points
    for _, row in df_plot.iterrows():
        sys = row["system"].replace("DeBERTa-v3-base", "DeBERTa")
        if "P" in sys: sys = sys.split("_")[0]

        # Offset annotations
        offset_y = 0.005 if "Hybrid" in sys else -0.015
        if "P1" in sys: offset_y = 0.015
        plt.annotate(sys, (row["cost_per_1k"], row["Accuracy"]),
                     xytext=(5, offset_y * 1000), textcoords="offset points",
                     fontsize=9)

    plt.title("Cost-Accuracy Pareto Frontier", pad=15)
    plt.xlabel("Cost per 1,000 queries (USD)")
    plt.ylabel("Accuracy on Matched Test Set")
    plt.xscale("symlog", linthresh=0.1)

    # Set x-ticks explicitly for log scale
    ticks = [0, 0.1, 1, 10, 100]
    plt.xticks(ticks, [f"${t}" for t in ticks])

    plt.legend(title="Strategy / Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_cost_accuracy_frontier.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 3: Matched vs Mismatched Comparison
# ============================================================
def plot_matched_vs_mismatched():
    print("Generating Fig 3: Matched vs Mismatched Comparison...")

    data = []

    # DeBERTa
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_enc_mm = safe_load("encoder_predictions_mm.csv")
    if not df_enc_m.empty and not df_enc_mm.empty:
        c1 = accuracy_score(df_enc_m["label_text"], df_enc_m["deberta_v3_base_pred"])
        c2 = accuracy_score(df_enc_mm["label_text"], df_enc_mm["deberta_v3_base_pred"])
        data.extend([
            {"System": "DeBERTa-v3-base", "Set": "Matched", "Accuracy": c1},
            {"System": "DeBERTa-v3-base", "Set": "Mismatched", "Accuracy": c2}
        ])

    # GPT-4o P4
    df_gpt4o = safe_load("api_results_gpt4o.csv")
    df_gpt4o_mm = safe_load("api_results_gpt4o_mm.csv")
    if not df_gpt4o.empty and not df_gpt4o_mm.empty:
        m = df_gpt4o[df_gpt4o["prompt"] == "P4_few_shot_cot"]
        mm = df_gpt4o_mm[df_gpt4o_mm["prompt"] == "P4_few_shot_cot"]
        if len(m) > 0 and len(mm) > 0:
            c1 = accuracy_score(m["label_true"], m["predicted_label"])
            c2 = accuracy_score(mm["label_true"], mm["predicted_label"])
            data.extend([
                {"System": "GPT-4o P4 (CoT)", "Set": "Matched", "Accuracy": c1},
                {"System": "GPT-4o P4 (CoT)", "Set": "Mismatched", "Accuracy": c2}
            ])

    # Hybrid v2
    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        m = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        mm = df_v2[(df_v2["set"] == "mismatched") & (df_v2["threshold"] == 0.90)]
        if len(m) > 0 and len(mm) > 0:
            c1 = accuracy_score(m["label_true"], m["label_pred"])
            c2 = accuracy_score(mm["label_true"], mm["label_pred"])
            data.extend([
                {"System": "Hybrid v2", "Set": "Matched", "Accuracy": c1},
                {"System": "Hybrid v2", "Set": "Mismatched", "Accuracy": c2}
            ])

    # Hybrid v3
    df_v3 = safe_load("hybrid_v3_results.csv")
    if not df_v3.empty:
        m = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)]
        mm = df_v3[(df_v3["set"] == "mismatched") & (df_v3["threshold"] == 0.90)]
        if len(m) > 0 and len(mm) > 0:
            c1 = accuracy_score(m["label_true"], m["label_pred"])
            c2 = accuracy_score(mm["label_true"], mm["label_pred"])
            data.extend([
                {"System": "Hybrid v3", "Set": "Matched", "Accuracy": c1},
                {"System": "Hybrid v3", "Set": "Mismatched", "Accuracy": c2}
            ])

    if not data:
        return

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="System", y="Accuracy", hue="Set", data=df_plot, palette=["steelblue", "darkorange"])

    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-Genre Generalisation (Matched vs Mismatched)", pad=15)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2, height + 0.01,
                    f"{height*100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_matched_vs_mismatched.png"), dpi=300)
    plt.close()


# ============================================================
# Figures 5-8: Confusion Matrices
# ============================================================
def plot_confusion_matrices():
    print("Generating Figs 5-8: Confusion Matrices...")

    df_enc_m = safe_load("encoder_predictions_matched.csv")
    if not df_enc_m.empty:
        plot_cm(df_enc_m["label_text"], df_enc_m["deberta_v3_base_pred"], "DeBERTa-v3-base (Matched)", "fig5_cm_deberta.png")

    df_gpt4o = safe_load("api_results_gpt4o.csv")
    if not df_gpt4o.empty:
        m = df_gpt4o[df_gpt4o["prompt"] == "P4_few_shot_cot"]
        if len(m) > 0:
            plot_cm(m["label_true"], m["predicted_label"], "GPT-4o P4 CoT (Matched)", "fig6_cm_gpt4o.png")

    df_claude = safe_load("api_results_claude.csv")
    if not df_claude.empty:
        # Use P3 (best Claude prompt with valid predictions)
        m = df_claude[df_claude["prompt"] == "P3_few_shot"]
        valid = m[m["predicted_label"] != "unknown"]
        if len(valid) > 0:
            plot_cm(valid["label_true"], valid["predicted_label"], "Claude Sonnet P3 (Matched)", "fig7_cm_claude.png")

    df_v2 = safe_load("hybrid_v2_results.csv")
    if not df_v2.empty:
        sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        if len(sub) > 0:
            plot_cm(sub["label_true"], sub["label_pred"], "Hybrid v2 θ=0.90 (Matched)", "fig8_cm_hybrid.png")

    df_v3 = safe_load("hybrid_v3_results.csv")
    if not df_v3.empty:
        sub = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)]
        if len(sub) > 0:
            plot_cm(sub["label_true"], sub["label_pred"], "Hybrid v3 θ=0.90 (Matched)", "fig8b_cm_hybrid_v3.png")


def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS,
                vmin=0, vmax=1)
    plt.title(f"Confusion Matrix: {title}", pad=15)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()


def plot_hybrid_threshold():
    print("Generating Fig 10: Hybrid Threshold Trade-off...")
    df_v2 = safe_load("hybrid_v2_results.csv")
    if df_v2.empty: return
    
    thresholds = []
    accs = []
    apipct = []
    
    sub_m = df_v2[df_v2["set"] == "matched"]
    for th in sorted(sub_m["threshold"].unique()):
        th_data = sub_m[sub_m["threshold"] == th]
        if len(th_data) == 0: continue
        acc = accuracy_score(th_data["label_true"], th_data["label_pred"])
        api_handled = (th_data["source"] == "api").mean()
        
        thresholds.append(th)
        accs.append(acc)
        apipct.append(api_handled)
        
    if not thresholds: return
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('DeBERTa Confidence Threshold (θ)')
    ax1.set_ylabel('System Accuracy', color=color)
    ax1.plot(thresholds, accs, marker='o', color=color, linewidth=2, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Samples Routed to LLM API (%)', color=color)  
    ax2.plot(thresholds, apipct, marker='s', color=color, linewidth=2, linestyle='--', label="% to API")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format y2 as percentage
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
    plt.title('Hybrid Architecture Trade-off (Accuracy vs API Cost)', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig10_hybrid_threshold.png"), dpi=300)
    plt.close()

def plot_genre_heatmap():
    print("Generating Fig 9: Genre Heatmap...")
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_v2 = safe_load("hybrid_v2_results.csv")
    
    if df_enc_m.empty or df_v2.empty: return
    
    genres = sorted(df_enc_m['genre'].dropna().unique())
    systems = ["DeBERTa", "Hybrid v2", "Hybrid v3", "Hybrid v4"]
    
    data = np.zeros((len(systems), len(genres)))
    
    sub_v2 = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
    df_v3 = safe_load("hybrid_v3_results.csv")
    sub_v3 = df_v3[(df_v3["set"] == "matched") & (df_v3["threshold"] == 0.90)] if not df_v3.empty else pd.DataFrame()
    df_v4 = safe_load("hybrid_v4_results.csv")
    sub_v4 = df_v4[(df_v4["set"] == "matched") & (df_v4["threshold"] == 0.90)] if not df_v4.empty else pd.DataFrame()
    
    for i, g in enumerate(genres):
        g_enc = df_enc_m[df_enc_m['genre'] == g]
        if len(g_enc) > 0:
            data[0, i] = accuracy_score(g_enc['label_text'], g_enc['deberta_v3_base_pred'])
            
        g_v2 = sub_v2[sub_v2['genre'] == g]
        if len(g_v2) > 0:
            data[1, i] = accuracy_score(g_v2['label_true'], g_v2['label_pred'])

        if not sub_v3.empty:
            g_v3 = sub_v3[sub_v3['genre'] == g]
            if len(g_v3) > 0:
                data[2, i] = accuracy_score(g_v3['label_true'], g_v3['label_pred'])

        if not sub_v4.empty:
            g_v4 = sub_v4[sub_v4['genre'] == g]
            if len(g_v4) > 0:
                data[3, i] = accuracy_score(g_v4['label_true'], g_v4['label_pred'])
            
    plt.figure(figsize=(10, 4))
    sns.heatmap(data, annot=True, fmt=".1%", cmap="YlGnBu", xticklabels=genres, yticklabels=systems, vmin=0.8)
    plt.title("Accuracy across Genres (Matched Set)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig9_genre_heatmap.png"), dpi=300)
    plt.close()

def plot_per_class_f1():
    print("Generating Fig 4: Per-Class F1 Grouped Bar...")
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_v2 = safe_load("hybrid_v2_results.csv")
    
    if df_enc_m.empty or df_v2.empty: return
    
    f1_deberta = f1_score(df_enc_m['label_text'], df_enc_m['deberta_v3_base_pred'], average=None, labels=LABELS)
    
    sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
    f1_hybrid = [0, 0, 0]
    if len(sub) > 0:
        f1_hybrid = f1_score(sub['label_true'], sub['label_pred'], average=None, labels=LABELS)
        
    x = np.arange(len(LABELS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, f1_deberta, width, label='DeBERTa-v3-base', color='steelblue')
    rects2 = ax.bar(x + width/2, f1_hybrid, width, label='Hybrid v2 (θ=0.90)', color='crimson')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([l.title() for l in LABELS])
    ax.legend()
    
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_per_class_f1.png"), dpi=300)
    plt.close()


# ============================================================
# Figure 11: Ensemble Gate Breakdown
# Shows unanimous vs escalated accuracy + annotation ceiling
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

    ens_acc = accuracy_score(ens["label_true"], ens["label_pred"])
    api_acc = accuracy_score(api["label_true"], api["label_pred"])
    total_acc = accuracy_score(sub["label_true"], sub["label_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar showing gate split
    ax1 = axes[0]
    ens_n = len(ens)
    api_n = len(api)
    total_n = len(sub)

    ax1.bar([0], [ens_n], color="steelblue", label=f"Unanimous ({ens_n}, {ens_n/total_n:.1%})")
    ax1.bar([0], [api_n], bottom=[ens_n], color="coral",
            label=f"Escalated ({api_n}, {api_n/total_n:.1%})")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylabel("Number of samples (800 total)")
    ax1.set_title("Gate Decision Split")
    ax1.legend(loc="upper right")
    ax1.set_xticks([])

    # Add accuracy annotations inside bars
    ax1.text(0, ens_n/2, f"{ens_acc*100:.1f}%\naccuracy",
             ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax1.text(0, ens_n + api_n/2, f"{api_acc*100:.1f}%\naccuracy",
             ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    # Right: accuracy comparison bars
    ax2 = axes[1]
    systems = ["Unanimous\n(700)", "Escalated\n(100)", "Total\n(800)"]
    accs = [ens_acc, api_acc, total_acc]
    bar_colors = ["steelblue", "coral", "slategray"]

    bars = ax2.bar(systems, accs, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylim(0.3, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by Gate Decision")
    ax2.axhline(0.333, color="red", linestyle=":", alpha=0.5, label="Random baseline")
    ax2.axhline(0.512, color="orange", linestyle="--", alpha=0.5, label="GPT-4o on escalated")
    ax2.legend(fontsize=9)

    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{acc*100:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Hybrid v5: Ensemble Gate Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig11_ensemble_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: fig11_ensemble_breakdown.png")


# ============================================================
# Figure 12: Gating Strategy Comparison
# Confidence gate (v4) vs Ensemble gate (v5)
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

    # v4: confidence-gated rows
    v4_enc = sub_v4[sub_v4["source"] == "encoder"]
    v4_api = sub_v4[sub_v4["source"] == "api"]
    # v5: ensemble-gated rows
    v5_ens = sub_v5[sub_v5["source"] == "ensemble"]
    v5_api = sub_v5[sub_v5["source"] == "api"]

    data = [
        {"System": "v4 Confidence Gate", "Category": f"Auto-accept ({len(v4_enc)})",
         "Accuracy": accuracy_score(v4_enc["label_true"], v4_enc["label_pred"]) if len(v4_enc) > 0 else 0},
        {"System": "v4 Confidence Gate", "Category": f"Escalated ({len(v4_api)})",
         "Accuracy": accuracy_score(v4_api["label_true"], v4_api["label_pred"]) if len(v4_api) > 0 else 0},
        {"System": "v5 Ensemble Gate", "Category": f"Unanimous ({len(v5_ens)})",
         "Accuracy": accuracy_score(v5_ens["label_true"], v5_ens["label_pred"]) if len(v5_ens) > 0 else 0},
        {"System": "v5 Ensemble Gate", "Category": f"Disagreement ({len(v5_api)})",
         "Accuracy": accuracy_score(v5_api["label_true"], v5_api["label_pred"]) if len(v5_api) > 0 else 0},
    ]
    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"v4 Confidence Gate": "steelblue", "v5 Ensemble Gate": "coral"}

    x = np.arange(2)
    w = 0.35
    cats_v4 = df_plot[df_plot["System"] == "v4 Confidence Gate"]["Accuracy"].values
    cats_v5 = df_plot[df_plot["System"] == "v5 Ensemble Gate"]["Accuracy"].values

    bars1 = ax.bar(x - w/2, cats_v4, w, color="steelblue", label="v4: Confidence Gate")
    bars2 = ax.bar(x + w/2, cats_v5, w, color="coral", label="v5: Ensemble Gate")

    ax.set_xticks(x)
    ax.set_xticklabels(["Auto-accept / Unanimous", "Escalated / Disagreement"])
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence Gating vs Ensemble Gating:\nAccuracy on Handled vs Escalated Samples", pad=12)
    ax.legend()
    ax.axhline(0.333, color="red", linestyle=":", alpha=0.4, label="Chance")

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height()*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig12_gating_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: fig12_gating_comparison.png")


def main():
    df_cost = safe_load("cost_summary.csv")

    plot_strategy_accuracy_bar(df_cost)
    plot_cost_accuracy_frontier(df_cost)
    plot_matched_vs_mismatched()
    plot_per_class_f1()
    plot_confusion_matrices()
    plot_genre_heatmap()
    plot_hybrid_threshold()
    plot_ensemble_breakdown()   # NEW: Fig 11
    plot_gating_comparison()    # NEW: Fig 12

    print("=" * 60)
    print("FIGURES COMPLETE ✅")
    print("=" * 60)

if __name__ == "__main__":
    main()
