#!/usr/bin/env python3
"""
Notebook 05d — Hybrid v5: 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT)
======================================================================
Gate logic: Run DeBERTa-v3-small, DeBERTa-v3-base, DeBERTa-v3-large in parallel.
  - If ALL THREE agree → use their unanimous prediction (free, ~95% accurate)
  - If ANY disagree   → escalate to GPT-4o P4 (CoT few-shot)

Why this works:
  - 700/800 samples (87.5%) are unanimous → 95.00% accurate, zero API cost
  - Only 100 samples (12.5%) escalate → LLM handles genuinely hard cases
  - Expected total: ~93-94% matched accuracy

This is fundamentally different from confidence-threshold gating (v1-v4):
  - Threshold gates use ONE encoder's uncertainty as the signal
  - Ensemble gates use DISAGREEMENT between models as the signal
  - Disagreement is a much stronger signal for genuine ambiguity

Prerequisite:
  02_encoder_baselines.py must have been run with all 5 models.
  encoder_predictions_matched.csv must contain all deberta columns.

Outputs:
  results/hybrid_v5_results.csv
"""

import os
import re
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from collections import Counter

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEB_MODELS = ["deberta_v3_small", "deberta_v3_base", "deberta_v3_large"]

# ============================================================
# Label parser — CoT-aware
# GPT-4o CoT responses end with "Label: <word>" after reasoning.
# Parser priority:
#   1. Look for explicit "label: <word>" anywhere in text (CoT format)
#   2. Check if first line starts with a label word (direct format)
#   3. Fallback: find last occurrence of a label word (avoids picking
#      up label words from the reasoning body)
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    text_lower = text.lower()

    # Priority 1: explicit "label:" marker (CoT output format)
    label_match = re.search(r'label\s*:\s*(contradiction|entailment|neutral)', text_lower)
    if label_match:
        return label_match.group(1)

    # Priority 2: first line starts with the label (direct format)
    first_line = text_lower.strip().split("\n")[0].strip()
    first_line_clean = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line_clean.startswith(label):
            return label

    # Priority 3: last occurrence of a label word in full text
    # (reasoning comes before conclusion, so last match is the answer)
    last_pos = -1
    last_label = "unknown"
    for label in ["contradiction", "entailment", "neutral"]:
        pos = text_lower.rfind(label)
        if pos > last_pos:
            last_pos = pos
            last_label = label
    return last_label


# ============================================================
# GPT-4o P4 (CoT few-shot) — same prompt as in notebook 03
# ============================================================
PROMPT_P4 = (
    "Classify the natural language inference relationship step by step.\n\n"
    "Examples:\n"
    'Premise: "The concert was held outdoors."\n'
    'Hypothesis: "The event took place inside a building."\n'
    "Step-by-step: The premise says outdoor; the hypothesis says inside. These directly contradict.\n"
    "Label: contradiction\n\n"
    'Premise: "She completed her PhD in linguistics."\n'
    'Hypothesis: "She has a doctoral degree."\n'
    "Step-by-step: A PhD is a doctoral degree. The hypothesis follows necessarily.\n"
    "Label: entailment\n\n"
    'Premise: "The report was published in March."\n'
    'Hypothesis: "The author spent years writing it."\n'
    "Step-by-step: Publication date says nothing about how long writing took.\n"
    "Label: neutral\n\n"
    "Now classify:\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Step-by-step:"
)

def call_gpt4o_p4(premise, hypothesis, max_retries=3):
    """GPT-4o with CoT few-shot (P4 prompt)."""
    from openai import OpenAI
    client = OpenAI()

    prompt = PROMPT_P4.format(premise=premise, hypothesis=hypothesis)
    INPUT_COST  = 2.50   # $ per 1M tokens
    OUTPUT_COST = 10.00

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,  # CoT needs more tokens than direct answer
                seed=42,
            )
            raw   = response.choices[0].message.content.strip()
            label = parse_label(raw)
            usage = response.usage
            cost  = (usage.prompt_tokens  * INPUT_COST  / 1_000_000
                   + usage.completion_tokens * OUTPUT_COST / 1_000_000)
            return label, usage.prompt_tokens + usage.completion_tokens, cost
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  GPT-4o error (attempt {attempt+1}): {e}  — retrying in {wait}s")
            time.sleep(wait)

    return "unknown", 0, 0.0


# ============================================================
# Core hybrid v5 runner
# ============================================================
def run_hybrid_v5(df_test, df_encoder, call_api_fn, set_name="matched"):
    """
    Hybrid v5: 3-DeBERTa ensemble gate.
    - Unanimous: use prediction directly (no API cost)
    - Split: escalate to LLM
    """
    results   = []
    api_calls = 0
    total_cost = 0.0

    pred_cols = [f"{m}_pred" for m in DEB_MODELS]
    conf_cols = [f"{m}_conf" for m in DEB_MODELS]

    for i in tqdm(range(len(df_test)), desc=f"Hybrid v5 [{set_name}]"):
        row     = df_test.iloc[i]
        enc_row = df_encoder.iloc[i]

        preds = [enc_row[col] for col in pred_cols]
        confs = [float(enc_row[col]) for col in conf_cols]

        unique_preds = set(preds)

        if len(unique_preds) == 1:
            # All 3 DeBERTas agree — unanimous
            final_pred = preds[0]
            avg_conf   = float(np.mean(confs))
            source     = "ensemble"
            tokens     = 0
            cost       = 0.0
        else:
            # Disagreement — escalate to LLM
            pred, tokens, cost = call_api_fn(row["premise"], row["hypothesis"])
            final_pred = pred
            avg_conf   = float(np.mean(confs))
            source     = "api"
            api_calls += 1
            total_cost += cost
            time.sleep(0.05)  # rate limit

        results.append({
            "idx"       : i,
            "hybrid"    : "v5_ensemble_gate",
            "set"       : set_name,
            "premise"   : row["premise"],
            "hypothesis": row["hypothesis"],
            "genre"     : row["genre"],
            "label_true": row["label_text"],
            "label_pred": final_pred,
            "source"    : source,
            "avg_conf"  : avg_conf,
            "deb_preds" : "|".join(preds),
            "tokens"    : tokens,
            "cost_usd"  : cost,
        })

    df_results = pd.DataFrame(results)

    y_true = df_results["label_true"]
    y_pred = df_results["label_pred"]
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro",
                      labels=["entailment", "neutral", "contradiction"])

    ensemble_pct = (df_results["source"] == "ensemble").mean() * 100
    api_pct      = (df_results["source"] == "api").mean() * 100
    errors       = (y_true != y_pred).sum()

    print(f"\n  ── Hybrid v5 | {set_name} ──────────────────────────────")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  Ensemble : {ensemble_pct:.1f}% (unanimous, free)")
    print(f"  API      : {api_pct:.1f}% ({api_calls} calls)")
    print(f"  Errors   : {errors}")
    print(f"  Total cost: ${total_cost:.4f}  (${total_cost/len(df_results)*1000:.4f}/1k)")

    # Per-genre breakdown
    print(f"\n  Per-genre accuracy:")
    for genre, grp in df_results.groupby("genre"):
        g_acc = accuracy_score(grp["label_true"], grp["label_pred"])
        g_api = (grp["source"] == "api").mean() * 100
        print(f"    {genre:15s}: {g_acc*100:.1f}%  (api={g_api:.0f}%)")

    return df_results, {
        "hybrid"       : "v5_ensemble_gate",
        "set"          : set_name,
        "accuracy"     : acc,
        "macro_f1"     : f1,
        "ensemble_pct" : ensemble_pct,
        "api_pct"      : api_pct,
        "api_calls"    : api_calls,
        "total_cost"   : total_cost,
        "cost_per_1k"  : total_cost / len(df_results) * 1000,
        "errors"       : int(errors),
    }


def main():
    # ── Load data ────────────────────────────────────────────
    df_test_m  = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))

    enc_m_path  = os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv")
    enc_mm_path = os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv")

    if not os.path.exists(enc_m_path):
        print("❌ Run 02_encoder_baselines.py first!")
        return

    df_enc_m  = pd.read_csv(enc_m_path)
    df_enc_mm = pd.read_csv(enc_mm_path) if os.path.exists(enc_mm_path) else None

    # ── Check all DeBERTa columns present ────────────────────
    required = [f"{m}_pred" for m in DEB_MODELS] + [f"{m}_conf" for m in DEB_MODELS]
    missing = [c for c in required if c not in df_enc_m.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print("   Re-run 02_encoder_baselines.py with all 5 models.")
        return

    # ── Preview gate statistics ───────────────────────────────
    unanimous = sum(
        1 for _, row in df_enc_m.iterrows()
        if len(set(row[f"{m}_pred"] for m in DEB_MODELS)) == 1
    )
    print(f"\n📊 Gate preview (matched):")
    print(f"   Unanimous (free): {unanimous}/800 = {unanimous/800*100:.1f}%")
    print(f"   Escalate to LLM:  {800-unanimous}/800 = {(800-unanimous)/800*100:.1f}%")
    print(f"   Expected LLM API calls: ~{800-unanimous}")

    print("\n" + "#" * 65)
    print("# HYBRID v5: 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT)")
    print("#" * 65)

    all_results = []
    all_metrics = []

    # Run on matched
    df_res, metrics = run_hybrid_v5(df_test_m, df_enc_m, call_gpt4o_p4,
                                    set_name="matched")
    all_results.append(df_res)
    all_metrics.append(metrics)

    # Run on mismatched
    if df_enc_mm is not None:
        missing_mm = [c for c in required if c not in df_enc_mm.columns]
        if not missing_mm:
            df_res, metrics = run_hybrid_v5(df_test_mm, df_enc_mm, call_gpt4o_p4,
                                            set_name="mismatched")
            all_results.append(df_res)
            all_metrics.append(metrics)
        else:
            print(f"\n⚠️  Skipping mismatched — missing columns: {missing_mm}")

    # ── Save ─────────────────────────────────────────────────
    df_all  = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, "hybrid_v5_results.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "#" * 65)
    print("# HYBRID v5 SUMMARY")
    print("#" * 65)
    df_metrics = pd.DataFrame(all_metrics)
    print(df_metrics[["set","accuracy","macro_f1","ensemble_pct","api_pct",
                       "api_calls","cost_per_1k","errors"]].to_string(index=False))

    best = df_metrics.loc[df_metrics["accuracy"].idxmax()]
    print(f"\n🏆 Best: {best['set']} → {best['accuracy']*100:.2f}% "
          f"| API={best['api_pct']:.1f}% | cost=${best['cost_per_1k']:.4f}/1k")

    print("\n" + "=" * 65)
    print("HYBRID v5 COMPLETE ✅")
    print("=" * 65)


if __name__ == "__main__":
    main()
