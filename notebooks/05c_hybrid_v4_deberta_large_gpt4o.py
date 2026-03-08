#!/usr/bin/env python3
"""
Notebook 05c — Hybrid v4 (DeBERTa-v3-large + GPT-4o)
=====================================================
Uses the SOTA DeBERTa-v3-large encoder (~91.8% MNLI) as the gatekeeper.
Uncertain samples (confidence < θ) are escalated to GPT-4o.

Thresholds: 0.85, 0.90, 0.95
Logic:
    if max_softmax_conf >= θ  → accept DeBERTa-v3-large label
    else                      → call GPT-4o P3 (few-shot)

Outputs:
    results/hybrid_v4_results.csv
"""

import os, re, time
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

def parse_label(text):
    if not text: return "unknown"
    first_line = text.strip().split("\n")[0].strip().lower()
    first_line = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label): return label
    text_clean = text.lower()
    for label in ["contradiction", "entailment", "neutral"]:
        if label in text_clean: return label
    return "unknown"

def call_gpt4o(premise, hypothesis, max_retries=3):
    from openai import OpenAI
    client = OpenAI()
    
    prompt = (
        "Classify the NLI relationship.\n\n"
        "Examples:\n"
        'Premise: "The concert was held outdoors."\n'
        'Hypothesis: "The event took place inside a building." → contradiction\n\n'
        'Premise: "She completed her PhD in linguistics."\n'
        'Hypothesis: "She has a doctoral degree." → entailment\n\n'
        'Premise: "The report was published in March."\n'
        'Hypothesis: "The author spent years writing it." → neutral\n\n'
        "Now classify:\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Label:"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
                seed=42
            )
            raw = response.choices[0].message.content.strip()
            label = parse_label(raw)
            usage = response.usage
            cost = (usage.prompt_tokens * 2.50 + usage.completion_tokens * 10.00) / 1_000_000
            return label, usage.prompt_tokens + usage.completion_tokens, cost
        except Exception as e:
            time.sleep(2 ** (attempt + 1))
            print(f"  API Error: {e}")
    return "unknown", 0, 0.0

def main():
    # Load test sets
    df_test_m = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))
    
    # Load encoder predictions
    enc_m = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv"))
    enc_mm = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv"))
    
    all_rows = []
    
    print("\n" + "#"*60)
    print("# RUNNING HYBRID v4 (DeBERTa-v3-large + GPT-4o)")
    print("#"*60)

    for th in THRESHOLDS:
        # Matched
        api_count = 0
        for i in tqdm(range(len(df_test_m)), desc=f"th={th} (matched)"):
            row = df_test_m.iloc[i]
            e_pred = enc_m.iloc[i]["deberta_v3_large_pred"]
            e_conf = enc_m.iloc[i]["deberta_v3_large_conf"]
            
            if e_conf >= th:
                final_pred, source, tokens, cost = e_pred, "encoder", 0, 0.0
            else:
                final_pred, tokens, cost = call_gpt4o(row["premise"], row["hypothesis"])
                source = "api"
                api_count += 1
                time.sleep(0.05)
            
            all_rows.append({
                "idx": i, "threshold": th, "set": "matched", "source": source,
                "label_true": row["label_text"], "label_pred": final_pred,
                "cost_usd": cost, "tokens": tokens, "genre": row["genre"]
            })
        print(f"  Matched th={th}: API handled {api_count/800:.1%}")

    # Mismatched at 0.90
    api_count = 0
    for i in tqdm(range(len(df_test_mm)), desc="th=0.90 (mismatched)"):
        row = df_test_mm.iloc[i]
        e_pred = enc_mm.iloc[i]["deberta_v3_large_pred"]
        e_conf = enc_mm.iloc[i]["deberta_v3_large_conf"]
        
        if e_conf >= 0.90:
            final_pred, source, tokens, cost = e_pred, "encoder", 0, 0.0
        else:
            final_pred, tokens, cost = call_gpt4o(row["premise"], row["hypothesis"])
            source = "api"
            api_count += 1
            time.sleep(0.05)
            
        all_rows.append({
            "idx": i, "threshold": 0.90, "set": "mismatched", "source": source,
            "label_true": row["label_text"], "label_pred": final_pred,
            "cost_usd": cost, "tokens": tokens, "genre": row["genre"]
        })
    print(f"  Mismatched th=0.90: API handled {api_count/400:.1%}")

    df_out = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "hybrid_v4_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    main()
