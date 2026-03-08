#!/usr/bin/env python3
"""
Notebook 03 — GPT-4o Prompting
================================
Runs 4 prompt strategies (P1–P4) on matched (800) and mismatched (400) test sets
using GPT-4o. Tracks token usage and cost per query.

Prompts:
    P1: Zero-shot simple
    P2: Zero-shot + definitions
    P3: Few-shot (3 balanced examples)
    P4: Few-shot + CoT (hidden reasoning)

Outputs:
    results/api_results_gpt4o.csv        — matched results (800 × 4 prompts)
    results/api_results_gpt4o_mm.csv     — mismatched results (400 × 4 prompts)
"""

import os
import re
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# Configuration
# ============================================================
load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL = "gpt-4o"
SEED = 42
TEMPERATURE = 0.0

# GPT-4o pricing (2026)
INPUT_COST_PER_M = 2.50
OUTPUT_COST_PER_M = 10.00

client = OpenAI()

# ============================================================
# Prompt Templates
# ============================================================
PROMPTS = {
    "P1_zero_shot": (
        "Classify the logical relationship between the premise and hypothesis.\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Respond with exactly one word: entailment, contradiction, or neutral."
    ),
    "P2_zero_shot_def": (
        "Definitions:\n"
        "- entailment: hypothesis is necessarily TRUE given the premise\n"
        "- contradiction: hypothesis is necessarily FALSE given the premise\n"
        "- neutral: truth of hypothesis cannot be determined from premise alone\n\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Respond with exactly one word: entailment, contradiction, or neutral."
    ),
    "P3_few_shot": (
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
    ),
    "P4_few_shot_cot": (
        "Classify NLI using step-by-step reasoning.\n\n"
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
        "Think step by step internally. Output only the final label.\n"
        "Label:"
    ),
    "P5_few_shot_32": "FUNCTION",  # Special marker for use with build_32shot_prompt
}

# 32-shot examples for P5 (hand-picked from dev set in notebook 09)
EXAMPLES_32 = [
    ("Try a selection at the Whisky Heritage Centre (they have over 100 for you to sample).", "There are at least 100 things to sample at Whisky Heritage Centre.", "entailment"),
    ("Until the late '60s, the Senate was deferential to the (many fewer) presidential nominees.", "The Senate was respectful of the presidential nominees.", "entailment"),
    ("INTEREST RATE - The price charged per unit of money borrowed per year, or other unit of time, usually expressed as a percentage.", "Interest is almost always expressed in terms of percent.", "entailment"),
    ("you know like CODA comes out of your out of your pay and the credit union comes out of your pay so we don't have to do anything there and the rest of it as far as my salary goes i just have it automatically deposited in into our bank", "I set things up so that my salary automatically deposits into our bank.", "entailment"),
    ("After the execution of Guru Tegh Bahadur, his son, Guru Gobind Singh, exalted the faithful to be ever ready for armed defense.", "Guru Tegh Bahadur has a son named Guru Gobind Singh", "entailment"),
    ("SSA is also seeking statutory authority for additional tools to recover current overpayments.", "SSA wants the authority to recover underpayments.", "contradiction"),
    ("The 37 hectares (91 acres) of garden are set on lands above the Wag Wag River.", "The nine hundred acres of garden are set on the lands above the Nile River.", "contradiction"),
    ("She has exchanged a hollow life for a heightened life, and has tried to comprehend all its turns, get its possibilities.", "She has chose to live a hollow life.", "contradiction"),
    ("oh thank God i've never been to Midland", "I go to Midland every other weekend.", "contradiction"),
    ("Then he sobered.", "He was drunk.", "contradiction"),
    ("In our family we have two sons in public life.", "Having Two sons in public puts strains on our family's privacy.", "neutral"),
    ("The Romans built roads and established towns, including the towns of Palmaria (Palma) and Pollentia.", "Some of the original towns built by the Romans are still in existence.", "neutral"),
    ("These adaptations are not uniformly valued.", "The values always change", "neutral"),
    ("yeah uh-huh but we look at it sort of as an investment in the future too", "The results will not be noticeable until further down the line.", "neutral"),
    ("I'm not sentimental, you know.\", She paused.", "Everyone thinks she's sentimental.", "neutral"),
    ("do you think most states have that or", "In your opinion, do most states have that?", "entailment"),
    ("but you know they kids seem like when they get ten or twelve years old they fall out of that", "Many kids leave the Scouts when they are pre-teens.", "entailment"),
    ("It is truly an honour.", "They were humbled.", "entailment"),
    ("Arafat is also ailing and has no clear successor.", "Arafat is in bad health and does not have a person chosen to take his place.", "entailment"),
    ("Troyes is also a center for shopping, with two outlet centers selling both French and international designer-name fashions and home accessories.", "Troues had two outlet centers which sell clothes and home accessories.", "entailment"),
    ("Several security managers said that by participating in our study, they hoped to gain insights on how to improve their information security programs.", "The security managers in the study joined in order to see what we were doing wrong.", "contradiction"),
    ("well his knees were bothering him yeah", "He was in tip-top condition.", "contradiction"),
    ("Jerusalem was divided into east and west, under the control of Jordan and Israel respectively.", "Israel won the war and Jerusalem.", "contradiction"),
    ("How to Watch Washington Week in Review : Back to front.", "The only way to watch Washington Week in Review is from the start to the end, as anything else would be viewed.", "contradiction"),
    ("Although, in this case the equipment did not have to be erected adjacent to an operating boiler, the erection included demolishing and erecting a complete boiler island.", "Although it was unnecessary, some of the equipment was adjacent.", "neutral"),
    ("The governing statute provides that a committee consisting of the Comptroller General, the Speaker of the House and President Pro Tempore of the Senate recommend an individual to the President.", "The process is long and will be reformed in the coming years.", "neutral"),
    ("Cave 31 tries to emulate the style of the great Hindu temple on a much smaller scale, but the artists here were working on much harder rock and so abandoned their effort.", "Cave 31 ran into problems because it was made of harder rock and everyone was disappointed.", "neutral"),
    ("So let me draw a slightly different moral from the saga of beach volleyball. If, as Speaker Gingrich says, the price of volleyball is eternal freedom, still it may take a village to raise a volleyball net.", "Speaker Gingrich thinks there is a linear connection between volleyball and freedom.", "neutral"),
    ("San'doro didn't make it sound hypothetical, thought Jon.", "San'doro's words were hollow, and Jon knew the truth of that immediately.", "neutral"),
    ("no never heard of it", "He does not know what it is.", "neutral"),
    ("Many Greeks in Asia Minor were forced to leave their homes and brought an influence of eastern cadences with them.", "The poor Greeks shouldn't have had to leave their homes.", "neutral"),
    ("ENVIRONMENTAL PROTECTION AGENCY", "Agency which is responsible for the destruction of the environment.", "contradiction"),
]

def build_32shot_prompt(premise, hypothesis):
    prompt = "Classify the NLI relationship. Respond with exactly one word: entailment, neutral, or contradiction.\n\nExamples:\n"
    for p, h, l in EXAMPLES_32:
        prompt += f"Premise: {p}\nHypothesis: {h}\nLabel: {l}\n\n"
    prompt += f"Now classify:\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    return prompt


def parse_label(text):
    """Robust NLI label parser with fallback matching."""
    if not text:
        return "unknown"

    # Check first line first
    first_line = text.strip().split("\n")[0].strip().lower()
    first_line = re.sub(r"[^a-z]", " ", first_line).strip()

    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label

    # Fallback: search entire text
    text_clean = text.lower().replace("*", "").replace("_", "")
    for label in ["contradiction", "entailment", "neutral"]:
        if label in text_clean:
            return label

    return "unknown"


def call_gpt4o(premise, hypothesis, prompt_template, max_retries=3):
    """Call GPT-4o API with retry logic and token tracking."""
    if prompt_template == "FUNCTION":
        prompt = build_32shot_prompt(premise, hypothesis)
    else:
        prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=20, # P5 only needs a single word, same as P1-P3
                seed=SEED,
            )

            raw_text = response.choices[0].message.content.strip()
            label = parse_label(raw_text)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            cost = (
                prompt_tokens * INPUT_COST_PER_M / 1_000_000
                + completion_tokens * OUTPUT_COST_PER_M / 1_000_000
            )

            return {
                "raw_response": raw_text,
                "predicted_label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost,
            }

        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return {
        "raw_response": "ERROR",
        "predicted_label": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def run_prompts_on_dataset(df, output_path, set_name="matched"):
    """Run all 4 prompts on a dataset with resume logic."""
    # Load existing results for resume
    existing = None
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        print(f"  Resuming from {len(existing)} existing rows in {output_path}")

    results = []
    total_cost = 0.0

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Running {prompt_name} on {set_name} ({len(df)} samples)")
        print(f"{'='*60}")

        prompt_cost = 0.0
        unknowns = 0

        for idx in tqdm(range(len(df)), desc=prompt_name):
            row = df.iloc[idx]

            # Resume logic: skip if already processed
            if existing is not None:
                mask = (
                    (existing["prompt"] == prompt_name)
                    & (existing["idx"] == idx)
                )
                if mask.any():
                    # Re-add existing result
                    existing_row = existing[mask].iloc[0].to_dict()
                    results.append(existing_row)
                    prompt_cost += existing_row.get("cost_usd", 0)
                    if existing_row.get("predicted_label") == "unknown":
                        unknowns += 1
                    continue

            # Call API
            result = call_gpt4o(row["premise"], row["hypothesis"], prompt_template)

            result_row = {
                "idx": idx,
                "prompt": prompt_name,
                "set": set_name,
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "genre": row["genre"],
                "label_true": row["label_text"],
                "predicted_label": result["predicted_label"],
                "raw_response": result["raw_response"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "cost_usd": result["cost_usd"],
            }
            results.append(result_row)
            prompt_cost += result["cost_usd"]

            if result["predicted_label"] == "unknown":
                unknowns += 1

            # Save checkpoint every 50 rows
            if (idx + 1) % 50 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)

            # Rate limiting
            time.sleep(0.1)

        total_cost += prompt_cost
        print(f"  {prompt_name}: cost=${prompt_cost:.4f}, unknowns={unknowns}")

    # Save final results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path} ({len(df_results)} rows)")
    print(f"   Total cost: ${total_cost:.4f}")

    return df_results


def evaluate_results(df_results):
    """Evaluate and print metrics for all prompts."""
    print("\n" + "#" * 60)
    print("# EVALUATION RESULTS")
    print("#" * 60)

    labels = ["entailment", "neutral", "contradiction"]

    for prompt_name in PROMPTS.keys():
        mask = df_results["prompt"] == prompt_name
        subset = df_results[mask]

        if len(subset) == 0:
            continue

        y_true = subset["label_true"]
        y_pred = subset["predicted_label"]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
        per_class = f1_score(y_true, y_pred, average=None, labels=labels)

        unknowns = (y_pred == "unknown").sum()
        avg_tokens = subset["total_tokens"].mean()
        avg_cost = subset["cost_usd"].mean()
        total_cost = subset["cost_usd"].sum()

        print(f"\n--- {prompt_name} ({subset['set'].iloc[0]}) ---")
        print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Macro F1:    {f1:.4f}")
        for i, label in enumerate(labels):
            print(f"  {label:15s} F1: {per_class[i]:.4f}")
        print(f"  Unknowns:    {unknowns}")
        print(f"  Avg tokens:  {avg_tokens:.0f}")
        print(f"  Avg cost:    ${avg_cost:.5f}")
        print(f"  Total cost:  ${total_cost:.4f}")


def main():
    from sklearn.metrics import accuracy_score, f1_score

    # Load test sets
    df_matched = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_mismatched = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))

    # ========================================================
    # Run on MATCHED test set
    # ========================================================
    print("\n" + "#" * 60)
    print("# GPT-4o P1–P4 ON MATCHED TEST SET (800)")
    print("#" * 60)

    matched_path = os.path.join(RESULTS_DIR, "api_results_gpt4o.csv")
    df_matched_results = run_prompts_on_dataset(df_matched, matched_path, "matched")
    evaluate_results(df_matched_results)

    # ========================================================
    # Run on MISMATCHED test set
    # ========================================================
    print("\n" + "#" * 60)
    print("# GPT-4o P1–P4 ON MISMATCHED TEST SET (400)")
    print("#" * 60)

    mm_path = os.path.join(RESULTS_DIR, "api_results_gpt4o_mm.csv")
    df_mm_results = run_prompts_on_dataset(df_mismatched, mm_path, "mismatched")
    evaluate_results(df_mm_results)

    # ========================================================
    # Cross-set comparison
    # ========================================================
    print("\n" + "#" * 60)
    print("# MATCHED vs MISMATCHED COMPARISON (GPT-4o)")
    print("#" * 60)

    labels = ["entailment", "neutral", "contradiction"]
    for prompt_name in PROMPTS.keys():
        m_mask = df_matched_results["prompt"] == prompt_name
        mm_mask = df_mm_results["prompt"] == prompt_name

        m_sub = df_matched_results[m_mask]
        mm_sub = df_mm_results[mm_mask]

        if len(m_sub) == 0 or len(mm_sub) == 0:
            continue

        m_acc = accuracy_score(m_sub["label_true"], m_sub["predicted_label"])
        mm_acc = accuracy_score(mm_sub["label_true"], mm_sub["predicted_label"])
        drop = mm_acc - m_acc

        print(f"  {prompt_name:20s}: matched={m_acc:.4f}  mm={mm_acc:.4f}  drop={drop:+.4f}")

    print("\n" + "=" * 60)
    print("GPT-4o PROMPTING COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
