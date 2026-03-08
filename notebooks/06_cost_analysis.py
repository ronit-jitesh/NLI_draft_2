#!/usr/bin/env python3
"""
Notebook 06 — Token Usage & Cost Analysis
===========================================
Aggregates token usage and cost data from all model results.
Computes cost per 1,000 queries for every system.

Outputs:
    results/cost_summary.csv
"""

import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

def load_if_exists(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def main():
    print("=" * 60)
    print("TOKEN USAGE & COST ANALYSIS")
    print("=" * 60)

    cost_rows = []

    # ========================================================
    # Free models (encoders + random baseline)
    # ========================================================
    cost_rows.append({
        "system": "Random Baseline",
        "strategy": "—",
        "avg_tokens": 0,
        "cost_per_query": 0.0,
        "cost_per_1k": 0.0,
    })

    for encoder in ["BERT-base", "DeBERTa-v3-small", "RoBERTa-base", "DeBERTa-v3-base", "DeBERTa-v3-large"]:
        cost_rows.append({
            "system": encoder,
            "strategy": "Encoder (free)",
            "avg_tokens": 0,
            "cost_per_query": 0.0,
            "cost_per_1k": 0.0,
        })

    # ========================================================
    # GPT-4o (from api_results_gpt4o.csv)
    # ========================================================
    df_gpt4o = load_if_exists("api_results_gpt4o.csv")
    if df_gpt4o is not None:
        for prompt_name in df_gpt4o["prompt"].unique():
            subset = df_gpt4o[df_gpt4o["prompt"] == prompt_name]
            avg_tokens = subset["total_tokens"].mean()
            avg_cost = subset["cost_usd"].mean()
            cost_rows.append({
                "system": f"GPT-4o {prompt_name}",
                "strategy": prompt_name,
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # GPT-5
    # ========================================================
    df_gpt5 = load_if_exists("api_results_gpt5.csv")
    if df_gpt5 is not None:
        for prompt_name in df_gpt5["prompt"].unique():
            subset = df_gpt5[df_gpt5["prompt"] == prompt_name]
            avg_tokens = subset["total_tokens"].mean()
            avg_cost = subset["cost_usd"].mean()
            cost_rows.append({
                "system": f"GPT-5 {prompt_name}",
                "strategy": prompt_name,
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # Claude Sonnet
    # ========================================================
    df_claude = load_if_exists("api_results_claude.csv")
    if df_claude is not None:
        for prompt_name in df_claude["prompt"].unique():
            subset = df_claude[df_claude["prompt"] == prompt_name]
            # Exclude ERROR rows (cost_usd=0 from failed API calls) from average
            valid = subset[subset["cost_usd"] > 0]
            if valid.empty:
                print(f"  Skipping Claude {prompt_name}: all rows are ERROR/zero-cost")
                continue
            avg_tokens = valid["total_tokens"].mean()
            avg_cost = valid["cost_usd"].mean()
            cost_rows.append({
                "system": f"Claude Sonnet {prompt_name}",
                "strategy": prompt_name,
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # Llama 3.3 (Groq — free)
    # ========================================================
    df_llama = load_if_exists("api_results_llama.csv")
    if df_llama is not None:
        for prompt_name in df_llama["prompt"].unique():
            subset = df_llama[df_llama["prompt"] == prompt_name]
            avg_tokens = subset["total_tokens"].mean()
            cost_rows.append({
                "system": f"Llama 3.3 {prompt_name}",
                "strategy": prompt_name,
                "avg_tokens": avg_tokens,
                "cost_per_query": 0.0,
                "cost_per_1k": 0.0,
            })

    # ========================================================
    # Hybrid v1
    # ========================================================
    df_v1 = load_if_exists("hybrid_v1_results.csv")
    if df_v1 is not None:
        for threshold in [0.85, 0.90, 0.95]:
            subset = df_v1[(df_v1["threshold"] == threshold) & (df_v1["set"] == "matched")]
            if len(subset) > 0:
                avg_cost = subset["cost_usd"].mean()
                avg_tokens = subset["tokens"].mean()
                cost_rows.append({
                    "system": f"Hybrid v1 θ={threshold}",
                    "strategy": "DeBERTa+GPT-4o",
                    "avg_tokens": avg_tokens,
                    "cost_per_query": avg_cost,
                    "cost_per_1k": avg_cost * 1000,
                })

    # ========================================================
    # Hybrid v2
    # ========================================================
    df_v2 = load_if_exists("hybrid_v2_results.csv")
    if df_v2 is not None:
        for threshold in [0.85, 0.90, 0.95]:
            subset = df_v2[(df_v2["threshold"] == threshold) & (df_v2["set"] == "matched")]
            if len(subset) > 0:
                avg_cost = subset["cost_usd"].mean()
                avg_tokens = subset["tokens"].mean()
                star = " ★" if threshold == 0.90 else ""
                cost_rows.append({
                    "system": f"Hybrid v2 θ={threshold}{star}",
                    "strategy": "DeBERTa+Claude",
                    "avg_tokens": avg_tokens,
                    "cost_per_query": avg_cost,
                    "cost_per_1k": avg_cost * 1000,
                })

    # ========================================================
    # Hybrid v3
    # ========================================================
    df_v3 = load_if_exists("hybrid_v3_results.csv")
    if df_v3 is not None:
        for threshold in [0.85, 0.90, 0.95]:
            subset = df_v3[(df_v3["threshold"] == threshold) & (df_v3["set"] == "matched")]
            if len(subset) > 0:
                avg_cost = subset["cost_usd"].mean()
                avg_tokens = subset["tokens"].mean()
                cost_rows.append({
                    "system": f"Hybrid v3 θ={threshold}",
                    "strategy": "DeBERTa+GPT-4o 32s",
                    "avg_tokens": avg_tokens,
                    "cost_per_query": avg_cost,
                    "cost_per_1k": avg_cost * 1000,
                })

    # ========================================================
    # Hybrid v4
    # ========================================================
    df_v4 = load_if_exists("hybrid_v4_results.csv")
    if df_v4 is not None:
        for threshold in [0.85, 0.90, 0.95]:
            subset = df_v4[(df_v4["threshold"] == threshold) & (df_v4["set"] == "matched")]
            if len(subset) > 0:
                avg_cost = subset["cost_usd"].mean()
                avg_tokens = subset["tokens"].mean()
                cost_rows.append({
                    "system": f"Hybrid v4 θ={threshold}",
                    "strategy": "DeBERTa-large+GPT-4o",
                    "avg_tokens": avg_tokens,
                    "cost_per_query": avg_cost,
                    "cost_per_1k": avg_cost * 1000,
                })

    # ========================================================
    # Hybrid v5
    # ========================================================
    df_v5 = load_if_exists("hybrid_v5_results.csv")
    if df_v5 is not None:
        subset = df_v5[df_v5["set"] == "matched"]
        if len(subset) > 0:
            avg_cost = subset["cost_usd"].mean()
            avg_tokens = subset["tokens"].mean()
            cost_rows.append({
                "system": "Hybrid v5 (Ensemble Gate)",
                "strategy": "3-DeBERTa + GPT-4o CoT",
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # Hybrid v5b (tiered ensemble — no new API calls)
    # ========================================================
    df_v5b = load_if_exists("hybrid_v5b_results.csv")
    if df_v5b is not None:
        subset = df_v5b[df_v5b["set"] == "matched"]
        if len(subset) > 0:
            avg_cost = subset["cost_usd"].mean() if "cost_usd" in subset.columns else 0.0
            avg_tokens = subset["tokens"].mean() if "tokens" in subset.columns else 0.0
            cost_rows.append({
                "system": "Hybrid v5b (Tiered Ensemble)",
                "strategy": "3-DeBERTa tiered + GPT-4o",
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # Hybrid v5c (ensemble gate + Claude Sonnet)
    # ========================================================
    df_v5c = load_if_exists("hybrid_v5c_results.csv")
    if df_v5c is not None:
        subset = df_v5c[df_v5c["set"] == "matched"]
        if len(subset) > 0:
            avg_cost = subset["cost_usd"].mean()
            avg_tokens = subset["tokens"].mean() if "tokens" in subset.columns else 0.0
            cost_rows.append({
                "system": "Hybrid v5c (Ensemble + Claude)",
                "strategy": "3-DeBERTa + Claude CoT",
                "avg_tokens": avg_tokens,
                "cost_per_query": avg_cost,
                "cost_per_1k": avg_cost * 1000,
            })

    # ========================================================
    # Output
    # ========================================================
    df_cost = pd.DataFrame(cost_rows)

    # Print formatted table
    print("\n" + "-" * 80)
    print(f"{'System':<30} {'Strategy':<20} {'Avg Tokens':>10} {'Cost/Query':>12} {'Cost/1K':>10}")
    print("-" * 80)
    for _, row in df_cost.iterrows():
        print(f"{row['system']:<30} {row['strategy']:<20} {row['avg_tokens']:>10.0f} "
              f"${row['cost_per_query']:>10.5f} ${row['cost_per_1k']:>8.2f}")
    print("-" * 80)

    # Save
    cost_path = os.path.join(RESULTS_DIR, "cost_summary.csv")
    df_cost.to_csv(cost_path, index=False)
    print(f"\n✅ Saved: {cost_path}")

    print("\n" + "=" * 60)
    print("COST ANALYSIS COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
