#!/usr/bin/env python3
"""
Notebook 05f — Hybrid v5c: 3-DeBERTa Ensemble Gate + Claude Sonnet (CoT)
=========================================================================
Identical gate logic to v5 but uses Claude Sonnet as the LLM fallback
instead of GPT-4o. Tests whether Claude's stronger reasoning helps on
the 100 genuinely ambiguous disagreement rows.

Gate:
  Unanimous (700 rows, 87.5%) → free, 95.0% accurate
  Disagreement (100 rows, 12.5%) → Claude Sonnet CoT

Why Claude instead of GPT-4o:
  - Hard rows are neutrality/entailment boundary cases requiring careful reasoning
  - Claude tends to be more conservative (avoids over-predicting entailment)
  - Claude CoT outputs are cleaner for parsing
  - Cost: ~$0.0008/call vs $0.0004 GPT-4o — marginal difference on 100 calls

Expected: ~$0.08 total cost, ~55-62% on hard rows → ~90.5-91.1% total

Outputs:
  results/hybrid_v5c_results.csv
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
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEB_MODELS = ["deberta_v3_small", "deberta_v3_base", "deberta_v3_large"]


# ============================================================
# CoT-aware label parser (fixed version)
# Priority: label: <word> marker → first line → last occurrence
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    text_lower = text.lower()

    # Priority 1: explicit "label:" marker (CoT format)
    m = re.search(r'label\s*:\s*(contradiction|entailment|neutral)', text_lower)
    if m:
        return m.group(1)

    # Priority 2: first line starts with label
    first_line = re.sub(r"[^a-z]", " ",
                        text_lower.strip().split("\n")[0]).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label

    # Priority 3: last occurrence (reasoning body comes before conclusion)
    last_pos, last_label = -1, "unknown"
    for label in ["contradiction", "entailment", "neutral"]:
        pos = text_lower.rfind(label)
        if pos > last_pos:
            last_pos, last_label = pos, label
    return last_label


# ============================================================
# Claude Sonnet CoT caller
# Same prompt structure as v2 but with explicit Label: line
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert at Natural Language Inference (NLI). "
    "Given a premise and hypothesis, classify their relationship. "
    "Think step by step, then output the label on a line starting with 'Label:'"
)

USER_PROMPT = (
    "Examples:\n\n"
    'Premise: "The concert was held outdoors."\n'
    'Hypothesis: "The event took place inside a building."\n'
    "Reasoning: The premise says outdoor; the hypothesis says inside. Direct contradiction.\n"
    "Label: contradiction\n\n"
    'Premise: "She completed her PhD in linguistics."\n'
    'Hypothesis: "She has a doctoral degree."\n'
    "Reasoning: A PhD is a doctoral degree — the hypothesis necessarily follows.\n"
    "Label: entailment\n\n"
    'Premise: "The report was published in March."\n'
    'Hypothesis: "The author spent years writing it."\n'
    "Reasoning: Publication date says nothing about writing duration.\n"
    "Label: neutral\n\n"
    "Now classify:\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Reasoning:"
)

def call_claude_cot(premise, hypothesis, max_retries=3):
    """Claude Sonnet with explicit CoT + Label: output format."""
    import anthropic
    client = anthropic.Anthropic(timeout=20.0)

    INPUT_COST  = 3.00   # $ per 1M tokens (Sonnet 4.5)
    OUTPUT_COST = 15.00

    user_content = USER_PROMPT.format(premise=premise, hypothesis=hypothesis)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw   = response.content[0].text.strip()
            label = parse_label(raw)
            cost  = (response.usage.input_tokens  * INPUT_COST  / 1_000_000
                   + response.usage.output_tokens * OUTPUT_COST / 1_000_000)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return label, raw, tokens, cost
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Claude error (attempt {attempt+1}): {e} — retry in {wait}s")
            time.sleep(wait)

    return "unknown", "", 0, 0.0


# ============================================================
# Core runner
# ============================================================
def run_v5c(df_test, df_encoder, set_name="matched"):
    pred_cols = [f"{m}_pred" for m in DEB_MODELS]
    conf_cols = [f"{m}_conf" for m in DEB_MODELS]

    results    = []
    api_calls  = 0
    total_cost = 0.0
    unknowns   = 0

    for i in tqdm(range(len(df_test)), desc=f"v5c [{set_name}]"):
        row     = df_test.iloc[i]
        enc_row = df_encoder.iloc[i]

        preds = [enc_row[col] for col in pred_cols]
        confs = [float(enc_row[col]) for col in conf_cols]

        if len(set(preds)) == 1:
            # Unanimous — free
            final_pred = preds[0]
            source     = "ensemble"
            raw        = ""
            tokens     = 0
            cost       = 0.0
        else:
            # Disagreement — call Claude
            final_pred, raw, tokens, cost = call_claude_cot(
                row["premise"], row["hypothesis"]
            )
            source = "api"
            api_calls  += 1
            total_cost += cost
            if final_pred == "unknown":
                unknowns += 1
            time.sleep(0.1)   # Claude rate limit buffer

        results.append({
            "idx"        : i,
            "hybrid"     : "v5c_ensemble_claude",
            "set"        : set_name,
            "premise"    : row["premise"],
            "hypothesis" : row["hypothesis"],
            "genre"      : row["genre"],
            "label_true" : row["label_text"],
            "label_pred" : final_pred,
            "source"     : source,
            "deb_preds"  : "|".join(preds),
            "avg_conf"   : float(np.mean(confs)),
            "raw_response": raw[:200] if raw else "",  # save first 200 chars for debugging
            "tokens"     : tokens,
            "cost_usd"   : cost,
        })

    df_out = pd.DataFrame(results)
    n      = len(df_out)
    acc    = accuracy_score(df_out["label_true"], df_out["label_pred"])
    f1     = f1_score(df_out["label_true"], df_out["label_pred"],
                      average="macro",
                      labels=["entailment","neutral","contradiction"])
    errors  = (df_out["label_true"] != df_out["label_pred"]).sum()
    api_pct = api_calls / n * 100
    api_rows = df_out[df_out["source"]=="api"]
    api_acc  = accuracy_score(api_rows["label_true"], api_rows["label_pred"]) if len(api_rows) else 0

    print(f"\n  ── Hybrid v5c | {set_name} ──────────────────────────")
    print(f"  Accuracy       : {acc*100:.2f}%  (errors: {errors})")
    print(f"  Macro F1       : {f1:.4f}")
    print(f"  Ensemble rows  : {(df_out['source']=='ensemble').sum()} ({100-api_pct:.1f}%, free)")
    print(f"  Claude API rows: {api_calls} ({api_pct:.1f}%)")
    print(f"  Claude accuracy: {api_acc*100:.1f}% on {len(api_rows)} escalated rows")
    print(f"  Unknowns       : {unknowns}  (parse failures)")
    print(f"  Total cost     : ${total_cost:.4f}  (${total_cost/n*1000:.4f}/1k)")

    # Compare Claude vs GPT-4o on same rows (if v5 exists)
    v5_path = os.path.join(RESULTS_DIR, "hybrid_v5_results.csv")
    if os.path.exists(v5_path):
        df_v5 = pd.read_csv(v5_path)
        df_v5_api = df_v5[(df_v5["set"]==set_name) & (df_v5["source"]=="api")]
        gpt_acc = accuracy_score(df_v5_api["label_true"], df_v5_api["label_pred"])
        print(f"\n  HEAD-TO-HEAD on {len(api_rows)} escalated rows:")
        print(f"    GPT-4o:  {gpt_acc*100:.1f}%")
        print(f"    Claude:  {api_acc*100:.1f}%")
        winner = "Claude" if api_acc > gpt_acc else "GPT-4o" if api_acc < gpt_acc else "Tie"
        print(f"    Winner:  {winner}")

    # Per-genre
    print(f"\n  Per-genre ({set_name}):")
    for genre, grp in df_out.groupby("genre"):
        g_acc = accuracy_score(grp["label_true"], grp["label_pred"])
        g_api = (grp["source"]=="api").mean()*100
        print(f"    {genre:14s}: {g_acc*100:.1f}%  (api={g_api:.0f}%)")

    return df_out, {
        "hybrid"    : "v5c_ensemble_claude",
        "set"       : set_name,
        "accuracy"  : acc,
        "macro_f1"  : f1,
        "errors"    : int(errors),
        "api_calls" : api_calls,
        "api_acc"   : api_acc,
        "api_pct"   : api_pct,
        "unknowns"  : unknowns,
        "total_cost": total_cost,
        "cost_per_1k": total_cost/n*1000,
    }


def main():
    df_test_m  = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))
    df_enc_m   = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv"))
    df_enc_mm  = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv"))

    # Check columns
    required = [f"{m}_pred" for m in DEB_MODELS] + [f"{m}_conf" for m in DEB_MODELS]
    missing  = [c for c in required if c not in df_enc_m.columns]
    if missing:
        print(f"❌ Missing encoder columns: {missing}")
        print("   Re-run 02_encoder_baselines.py first.")
        return

    # Preview
    unanimous_m = sum(
        1 for _, r in df_enc_m.iterrows()
        if len(set(r[f"{m}_pred"] for m in DEB_MODELS)) == 1
    )
    print(f"\n📊 Gate preview:")
    print(f"   Matched  — unanimous: {unanimous_m}/800 ({unanimous_m/8:.1f}%)  "
          f"escalate: {800-unanimous_m} Claude calls")
    unanimous_mm = sum(
        1 for _, r in df_enc_mm.iterrows()
        if len(set(r[f"{m}_pred"] for m in DEB_MODELS)) == 1
    )
    print(f"   Mismatched — unanimous: {unanimous_mm}/400 ({unanimous_mm/4:.1f}%)  "
          f"escalate: {400-unanimous_mm} Claude calls")
    total_calls = (800-unanimous_m) + (400-unanimous_mm)
    print(f"   Total Claude calls: ~{total_calls}  (~${total_calls*0.0008:.2f})")

    print("\n" + "#"*60)
    print("# HYBRID v5c: Ensemble Gate + Claude Sonnet CoT")
    print("#"*60)

    all_results, all_metrics = [], []

    df_res, metrics = run_v5c(df_test_m, df_enc_m, "matched")
    all_results.append(df_res)
    all_metrics.append(metrics)

    if "deberta_v3_large_pred" in df_enc_mm.columns:
        df_res, metrics = run_v5c(df_test_mm, df_enc_mm, "mismatched")
        all_results.append(df_res)
        all_metrics.append(metrics)
    else:
        print("\n⚠️  Skipping mismatched — missing large model columns in encoder_predictions_mm.csv")

    # Save
    df_all   = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, "hybrid_v5c_results.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

    # Final summary
    print("\n" + "="*60)
    print("HYBRID v5c SUMMARY")
    print("="*60)
    print(f"{'Set':<12} {'Accuracy':>9} {'Claude%':>8} {'Claude acc':>11} {'Cost/1k':>9} {'Unknowns':>9}")
    print("-"*60)
    for m in all_metrics:
        print(f"  {m['set']:<10} {m['accuracy']*100:>8.2f}%  "
              f"{m['api_pct']:>7.1f}%  "
              f"{m['api_acc']*100:>10.1f}%  "
              f"${m['cost_per_1k']:>8.4f}  "
              f"{m['unknowns']:>8}")

    print("\n" + "="*60)
    print("COMPARISON: Claude vs GPT-4o as ensemble fallback")
    print("="*60)
    systems = [
        ("DeBERTa-v3-base",     90.12, 90.75, 0,   "$0.000"),
        ("Hybrid v1 θ=0.90",    90.12, 91.25, 3.8, "$0.013"),
        ("Hybrid v4 θ=0.90",    90.62, 90.50, 2.0, "$0.007"),
        ("Hybrid v5 (GPT-4o)",  89.50, 90.25, 12.5,"$0.288"),
    ]
    print(f"  {'System':<28} {'Matched':>8} {'MM':>7} {'API%':>6} {'Cost/1k':>9}")
    print("  " + "-"*56)
    for name, m, mm, api, cost in systems:
        print(f"  {name:<28} {m:.2f}%  {mm:.2f}%  {api:>5.1f}%  {cost:>8}")
    for m in all_metrics:
        if m['set'] == 'matched':
            mm_metrics = next((x for x in all_metrics if x['set']=='mismatched'), None)
            mm_acc = f"{mm_metrics['accuracy']*100:.2f}%" if mm_metrics else "—"
            print(f"  {'Hybrid v5c (Claude)':<28} {m['accuracy']*100:.2f}%  "
                  f"{mm_acc:>7}  {m['api_pct']:>5.1f}%  ${m['cost_per_1k']:>7.4f}")

    print("\n" + "="*60)
    print("HYBRID v5c COMPLETE ✅")
    print("="*60)


if __name__ == "__main__":
    main()
