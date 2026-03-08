#!/usr/bin/env python3
"""
Notebook 05 — Hybrid Gatekeeper
=================================
Implements confidence-gated hybrid systems:
    v1: DeBERTa-v3-large + GPT-4o P3 (few-shot)
    v2: DeBERTa-v3-large + Claude Sonnet P4 (CoT few-shot) — HEADLINE RESULT

Thresholds: 0.85, 0.90, 0.95
Runs on BOTH matched (800) and mismatched (400) test sets.

Architecture:
    Input → DeBERTa-v3-large → confidence ≥ θ? → YES → use encoder (free)
                                               → NO  → call LLM API

Outputs:
    results/hybrid_v1_results.csv
    results/hybrid_v2_results.csv
"""

import os
import re
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLDS = [0.85, 0.90, 0.95]

# ============================================================
# Label parser
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    first_line = text.strip().split("\n")[0].strip().lower()
    first_line = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label
    text_clean = text.lower().replace("*", "").replace("_", "")
    for label in ["contradiction", "entailment", "neutral"]:
        if label in text_clean:
            return label
    return "unknown"


# ============================================================
# GPT-4o P3 caller (for Hybrid v1)
# ============================================================
PROMPT_P3 = (
    "Classify the NLI relationship.\n\n"
    "Examples:\n"
    'Premise: "The concert was held outdoors."\n'
    'Hypothesis: "The event took place inside a building." → contradiction\n\n'
    'Premise: "She completed her PhD in linguistics."\n'
    'Hypothesis: "She has a doctoral degree." → entailment\n\n'
    'Premise: "The report was published in March."\n'
    'Hypothesis: "The author spent years writing it." → neutral\n\n'
    "Now classify:\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Label:"
)

def call_gpt4o_p3(premise, hypothesis, max_retries=3):
    """Call GPT-4o with few-shot prompt (P3)."""
    from openai import OpenAI
    client = OpenAI()

    prompt = PROMPT_P3.format(premise=premise, hypothesis=hypothesis)
    INPUT_COST = 2.50
    OUTPUT_COST = 10.00

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
                seed=42,
            )
            raw = response.choices[0].message.content.strip()
            label = parse_label(raw)
            usage = response.usage
            cost = (
                usage.prompt_tokens * INPUT_COST / 1_000_000
                + usage.completion_tokens * OUTPUT_COST / 1_000_000
            )
            return label, usage.prompt_tokens + usage.completion_tokens, cost
        except Exception as e:
            time.sleep(2 ** (attempt + 1))
            print(f"  GPT-4o error: {e}")

    return "unknown", 0, 0.0


# ============================================================
# Claude Sonnet P4 caller (for Hybrid v2)
# ============================================================
def call_claude_cot(premise, hypothesis, max_retries=3):
    """Call Claude Sonnet with CoT few-shot prompt."""
    import anthropic
    client = anthropic.Anthropic(
        timeout=15.0  # Force a timeout so the script doesn't hang forever
    )

    system_prompt = (
        "Classify NLI using step-by-step reasoning.\n\n"
        "Examples:\n"
        'Premise: "The concert was held outdoors."\n'
        'Hypothesis: "The event took place inside a building." → contradiction\n\n'
        'Premise: "She completed her PhD in linguistics."\n'
        'Hypothesis: "She has a doctoral degree." → entailment\n\n'
        'Premise: "The report was published in March."\n'
        'Hypothesis: "The author spent years writing it." → neutral\n\n'
    )
    user_prompt = (
        "Now classify:\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Output the label on the first line, then explain your reasoning briefly.\n"
        "Label:"
    )

    INPUT_COST = 3.00
    OUTPUT_COST = 15.00

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=100,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            label = parse_label(raw)
            input_tok = response.usage.input_tokens
            output_tok = response.usage.output_tokens
            cost = (
                input_tok * INPUT_COST / 1_000_000
                + output_tok * OUTPUT_COST / 1_000_000
            )
            return label, input_tok + output_tok, cost
        except Exception as e:
            time.sleep(2 ** (attempt + 1))
            print(f"  Claude error: {e}")

    return "unknown", 0, 0.0


# ============================================================
# Core hybrid runner
# ============================================================
def run_hybrid(df_test, df_encoder, call_api_fn, threshold,
               model_col="deberta_v3_large_pred",
               conf_col="deberta_v3_large_conf",
               set_name="matched", hybrid_name="v1"):
    """
    Run hybrid gatekeeper at a given threshold.

    Returns DataFrame with per-row results and summary metrics.
    """
    results = []
    api_calls = 0
    total_cost = 0.0

    for i in tqdm(range(len(df_test)), desc=f"Hybrid {hybrid_name} θ={threshold}"):
        row = df_test.iloc[i]
        enc_pred = df_encoder.iloc[i][model_col]
        enc_conf = df_encoder.iloc[i][conf_col]

        if enc_conf >= threshold:
            # Confident — use encoder (free)
            final_pred = enc_pred
            source = "encoder"
            tokens = 0
            cost = 0.0
        else:
            # Uncertain — call LLM
            pred, tokens, cost = call_api_fn(row["premise"], row["hypothesis"])
            final_pred = pred
            source = "api"
            api_calls += 1
            total_cost += cost
            time.sleep(0.05)  # Rate limiting

        results.append({
            "idx": i,
            "hybrid": hybrid_name,
            "threshold": threshold,
            "set": set_name,
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "genre": row["genre"],
            "label_true": row["label_text"],
            "label_pred": final_pred,
            "source": source,
            "enc_conf": enc_conf,
            "tokens": tokens,
            "cost_usd": cost,
        })

    df_results = pd.DataFrame(results)

    # Compute metrics
    y_true = df_results["label_true"]
    y_pred = df_results["label_pred"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro",
                  labels=["entailment", "neutral", "contradiction"])

    encoder_pct = (df_results["source"] == "encoder").mean() * 100
    api_pct = (df_results["source"] == "api").mean() * 100

    print(f"\n  Hybrid {hybrid_name} | θ={threshold} | {set_name}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Encoder:   {encoder_pct:.1f}% | API: {api_pct:.1f}%")
    print(f"  API calls: {api_calls} | Total cost: ${total_cost:.4f}")

    return df_results, {
        "hybrid": hybrid_name,
        "threshold": threshold,
        "set": set_name,
        "accuracy": acc,
        "macro_f1": f1,
        "encoder_pct": encoder_pct,
        "api_pct": api_pct,
        "api_calls": api_calls,
        "total_cost": total_cost,
    }


def main():
    # Load test data
    df_test_m = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))

    # Load encoder predictions (must run 02 first)
    enc_m_path = os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv")
    enc_mm_path = os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv")

    if not os.path.exists(enc_m_path):
        print("❌ Run notebook 02_encoder_baselines.py first!")
        return

    df_enc_m = pd.read_csv(enc_m_path)
    df_enc_mm = pd.read_csv(enc_mm_path) if os.path.exists(enc_mm_path) else None

    all_results_v1 = []
    all_results_v2 = []
    all_metrics = []

    # ========================================================
    # HYBRID v1: DeBERTa + GPT-4o P3
    # ========================================================
    print("\n" + "#" * 60)
    print("# HYBRID v1: DeBERTa-v3-base + GPT-4o P3 (Few-Shot)")
    print("#" * 60)

    for threshold in THRESHOLDS:
        # Matched
        df_res, metrics = run_hybrid(
            df_test_m, df_enc_m, call_gpt4o_p3, threshold,
            set_name="matched", hybrid_name="v1_deberta_gpt4o"
        )
        all_results_v1.append(df_res)
        all_metrics.append(metrics)

    # v1 on mismatched at θ=0.90
    if df_enc_mm is not None:
        df_res, metrics = run_hybrid(
            df_test_mm, df_enc_mm, call_gpt4o_p3, 0.90,
            set_name="mismatched", hybrid_name="v1_deberta_gpt4o"
        )
        all_results_v1.append(df_res)
        all_metrics.append(metrics)

    # Save v1
    df_v1 = pd.concat(all_results_v1, ignore_index=True)
    v1_path = os.path.join(RESULTS_DIR, "hybrid_v1_results.csv")
    df_v1.to_csv(v1_path, index=False)
    print(f"\n✅ Saved: {v1_path}")

    # ========================================================
    # HYBRID v2: DeBERTa + Claude Sonnet CoT  (HEADLINE)
    # ========================================================
    print("\n" + "#" * 60)
    print("# HYBRID v2: DeBERTa-v3-base + Claude Sonnet CoT ⭐")
    print("#" * 60)

    for threshold in THRESHOLDS:
        # Matched
        df_res, metrics = run_hybrid(
            df_test_m, df_enc_m, call_claude_cot, threshold,
            set_name="matched", hybrid_name="v2_deberta_claude"
        )
        all_results_v2.append(df_res)
        all_metrics.append(metrics)

    # v2 on mismatched at θ=0.90
    if df_enc_mm is not None:
        df_res, metrics = run_hybrid(
            df_test_mm, df_enc_mm, call_claude_cot, 0.90,
            set_name="mismatched", hybrid_name="v2_deberta_claude"
        )
        all_results_v2.append(df_res)
        all_metrics.append(metrics)

    # Save v2
    df_v2 = pd.concat(all_results_v2, ignore_index=True)
    v2_path = os.path.join(RESULTS_DIR, "hybrid_v2_results.csv")
    df_v2.to_csv(v2_path, index=False)
    print(f"\n✅ Saved: {v2_path}")

    # ========================================================
    # Summary
    # ========================================================
    print("\n" + "#" * 60)
    print("# HYBRID GATEKEEPER SUMMARY")
    print("#" * 60)

    df_metrics = pd.DataFrame(all_metrics)
    print("\n" + df_metrics.to_string(index=False))

    # Highlight best result
    best = df_metrics.loc[df_metrics["accuracy"].idxmax()]
    print(f"\n🏆 Best result: {best['hybrid']} θ={best['threshold']} ({best['set']})")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1: {best['macro_f1']:.4f}")
    print(f"   API calls: {best['api_calls']}")
    print(f"   Total cost: ${best['total_cost']:.4f}")

    print("\n" + "=" * 60)
    print("HYBRID GATEKEEPER COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
