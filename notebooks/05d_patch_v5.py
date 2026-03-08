#!/usr/bin/env python3
"""
05d_patch_v5.py — Re-run ONLY the unknown/failed API rows from hybrid_v5_results.csv
=======================================================================================
The original v5 run had 70/100 "unknown" predictions on the matched set and
37/51 on mismatched because the CoT parser was broken (checked first line only,
but GPT-4o CoT output puts the label at the END after reasoning).

This patch:
  1. Loads existing hybrid_v5_results.csv
  2. Finds all rows where source='api' AND label_pred='unknown'
  3. Re-calls GPT-4o with the FIXED parser (label: <word> regex priority)
  4. Overwrites those rows in the CSV
  5. Prints updated accuracy

Cost: ~107 rows × $0.0004 ≈ $0.04 total
"""

import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
V5_PATH     = os.path.join(RESULTS_DIR, "hybrid_v5_results.csv")


# ============================================================
# Fixed CoT-aware label parser
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    text_lower = text.lower()

    # Priority 1: explicit "label:" marker (CoT output)
    m = re.search(r'label\s*:\s*(contradiction|entailment|neutral)', text_lower)
    if m:
        return m.group(1)

    # Priority 2: first line starts with label (direct format)
    first_line = re.sub(r"[^a-z]", " ", text_lower.strip().split("\n")[0]).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label

    # Priority 3: last occurrence wins (reasoning body comes first)
    last_pos, last_label = -1, "unknown"
    for label in ["contradiction", "entailment", "neutral"]:
        pos = text_lower.rfind(label)
        if pos > last_pos:
            last_pos, last_label = pos, label
    return last_label


# ============================================================
# GPT-4o P4 caller (CoT, same prompt as 05d)
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
    from openai import OpenAI
    client = OpenAI()
    prompt = PROMPT_P4.format(premise=premise, hypothesis=hypothesis)
    INPUT_COST, OUTPUT_COST = 2.50, 10.00

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
                seed=42,
            )
            raw   = response.choices[0].message.content.strip()
            label = parse_label(raw)
            usage = response.usage
            cost  = (usage.prompt_tokens * INPUT_COST / 1_000_000
                   + usage.completion_tokens * OUTPUT_COST / 1_000_000)
            return label, raw, usage.prompt_tokens + usage.completion_tokens, cost
        except Exception as e:
            time.sleep(2 ** (attempt + 1))
            print(f"  Error: {e}")
    return "unknown", "", 0, 0.0


def main():
    if not os.path.exists(V5_PATH):
        print(f"❌ {V5_PATH} not found. Run 05d first.")
        return

    df = pd.read_csv(V5_PATH)
    total = len(df)

    # Find rows to fix: API calls that returned unknown
    mask = (df["source"] == "api") & (df["label_pred"] == "unknown")
    fix_idx = df[mask].index.tolist()

    print(f"Total rows:          {total}")
    print(f"Rows to re-run:      {len(fix_idx)}")
    print(f"  matched unknown:   {((df['set']=='matched') & mask).sum()}")
    print(f"  mismatched unknown:{((df['set']=='mismatched') & mask).sum()}")
    print(f"Estimated cost:      ~${len(fix_idx) * 0.00045:.3f}")
    print()

    if len(fix_idx) == 0:
        print("✅ No unknown rows found — nothing to fix.")
        return

    repaired = 0
    total_cost = 0.0

    for i in tqdm(fix_idx, desc="Re-running unknown rows"):
        row = df.loc[i]
        label, raw, tokens, cost = call_gpt4o_p4(row["premise"], row["hypothesis"])

        df.at[i, "label_pred"] = label
        df.at[i, "tokens"]     = tokens
        df.at[i, "cost_usd"]   = row["cost_usd"] + cost  # add re-run cost

        total_cost += cost
        if label != "unknown":
            repaired += 1
        time.sleep(0.05)

    # Save back
    df.to_csv(V5_PATH, index=False)
    print(f"\n✅ Saved: {V5_PATH}")
    print(f"   Repaired {repaired}/{len(fix_idx)} rows  (cost: ${total_cost:.4f})")

    # Print updated results
    print()
    print("=" * 60)
    print("UPDATED HYBRID v5 RESULTS")
    print("=" * 60)
    for s in ["matched", "mismatched"]:
        grp = df[df["set"] == s]
        if grp.empty:
            continue
        acc    = accuracy_score(grp["label_true"], grp["label_pred"])
        f1     = f1_score(grp["label_true"], grp["label_pred"],
                          average="macro",
                          labels=["entailment","neutral","contradiction"])
        n      = len(grp)
        errors = (grp["label_true"] != grp["label_pred"]).sum()
        api_n  = (grp["source"] == "api").sum()
        api_pct = api_n / n * 100
        cost_1k = grp["cost_usd"].sum() / n * 1000
        unknown = (grp["label_pred"] == "unknown").sum()

        print(f"\n  {s.upper()} (N={n})")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Macro F1  : {f1:.4f}")
        print(f"  Errors    : {errors}")
        print(f"  API%      : {api_pct:.1f}%  ({api_n} calls)")
        print(f"  Unknown   : {unknown}")
        print(f"  Cost/1k   : ${cost_1k:.4f}")

    print()
    print("=" * 60)
    print("PATCH COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
