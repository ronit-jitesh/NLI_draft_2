#!/usr/bin/env python3
"""
Notebook 04 — Other LLMs (GPT-5, Claude Sonnet 4.5, Llama 3.3 70B)
====================================================================
Runs the same 4 prompt strategies on matched test set (800 samples)
across 3 additional LLM providers.

Models:
    1. GPT-5        — OpenAI reasoning model (no temp/max_tokens)
    2. Claude Sonnet — Anthropic claude-sonnet-4-5
    3. Llama 3.3    — Groq free tier (llama-3.3-70b-versatile)

Outputs:
    results/api_results_gpt5.csv
    results/api_results_claude.csv
    results/api_results_llama.csv
"""

import os
import re
import time
import pandas as pd
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

SEED = 42

# Prompt templates (same as Notebook 03)
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
}


def parse_label(text):
    """Robust NLI label parser with fallback matching."""
    if not text:
        return "unknown"
    first_line = text.strip().split("\n")[0].strip().lower()
    first_line = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label

    # Try to find "Label: [LABEL]" pattern at the end
    text_clean = text.lower().replace("*", "").replace("_", "")
    matches = re.findall(r"label:\s*(entailment|neutral|contradiction)", text_clean)
    if matches:
        return matches[-1] # Return the last one found

    # Fallback: find any label mention, prioritising the end of the text
    for label in ["contradiction", "entailment", "neutral"]:
        if label in text_clean:
            # Check the last 100 chars first
            if label in text_clean[-100:]:
                return label
            return label
    return "unknown"


# ============================================================
# GPT-5 Client
# ============================================================
def call_gpt5(premise, hypothesis, prompt_template, max_retries=3):
    """Call o3-mini (reasoning model)."""
    from openai import OpenAI
    client = OpenAI()

    prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)

    INPUT_COST = 1.10
    OUTPUT_COST = 4.40

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=120,
            )

            raw_text = response.choices[0].message.content.strip()
            label = parse_label(raw_text)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            # Track reasoning tokens if available
            reasoning_tokens = 0
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

            cost = (
                prompt_tokens * INPUT_COST / 1_000_000
                + completion_tokens * OUTPUT_COST / 1_000_000
            )

            return {
                "raw_response": raw_text,
                "predicted_label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost,
            }

        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            print(f"  GPT-5 error (attempt {attempt+1}): {e}")
            time.sleep(wait_time)

    return {
        "raw_response": "ERROR",
        "predicted_label": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


# ============================================================
# Claude Sonnet 4.5 Client
# ============================================================
def call_claude(premise, hypothesis, prompt_template, max_retries=3):
    """Call Claude Sonnet 4.5 with max_tokens=100 (not 10!)."""
    import anthropic
    client = anthropic.Anthropic(timeout=30.0)  # increased from 15s to handle CoT

    prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)

    # Claude Sonnet pricing: $3 input / $15 output per 1M tokens
    INPUT_COST = 3.00
    OUTPUT_COST = 15.00

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=100,  # was 1000 — caused timeouts on P4 CoT; 100 is enough for reasoning + label
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text.strip()
            label = parse_label(raw_text)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (
                input_tokens * INPUT_COST / 1_000_000
                + output_tokens * OUTPUT_COST / 1_000_000
            )

            return {
                "raw_response": raw_text,
                "predicted_label": label,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": cost,
            }

        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            print(f"  Claude error (attempt {attempt+1}): {e}")
            time.sleep(wait_time)

    return {
        "raw_response": "ERROR",
        "predicted_label": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


# ============================================================
# Llama 3.3 70B (Groq) Client
# ============================================================
def call_llama_groq(premise, hypothesis, prompt_template, max_retries=3):
    """Call Llama 3.3 70B via Groq (OpenAI-compatible API)."""
    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {
            "raw_response": "NO_API_KEY",
            "predicted_label": "unknown",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    client = OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )

            raw_text = response.choices[0].message.content.strip()
            label = parse_label(raw_text)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            # Groq free tier: $0 cost
            return {
                "raw_response": raw_text,
                "predicted_label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": 0.0,
            }

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                # Hit rate limit — wait longer
                wait_time = 60
                print(f"  Groq rate limit hit — waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait_time = 2 ** (attempt + 1)
                print(f"  Llama/Groq error (attempt {attempt+1}): {e}")
                time.sleep(wait_time)

    return {
        "raw_response": "ERROR",
        "predicted_label": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


# ============================================================
# Generic runner with resume logic
# ============================================================
def run_model(df, call_fn, model_name, output_path, prompts_to_run=None):
    """Run a model with all prompts on a dataset with resume logic."""
    if prompts_to_run is None:
        prompts_to_run = list(PROMPTS.keys())

    # Load existing results for resume
    existing = None
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        print(f"  Resuming from {len(existing)} existing rows")

    # Initialize results from existing file to be non-destructive
    if existing is not None:
        results = existing.to_dict("records")
        print(f"  Initialized results list with {len(results)} existing records.")
    else:
        results = []
    
    total_cost = 0.0

    for prompt_name in prompts_to_run:
        prompt_template = PROMPTS[prompt_name]
        print(f"\n{'='*60}")
        print(f"Running {model_name} / {prompt_name} ({len(df)} samples)")
        print(f"{'='*60}")

        prompt_cost = 0.0
        unknowns = 0

        for idx in tqdm(range(len(df)), desc=f"{model_name}/{prompt_name}"):
            row = df.iloc[idx]

            # Check if this specific (model, prompt, idx) is already in results
            already_done = False
            if results:
                # Optimized check: look at the last few thousand rows or use a set if needed
                # For 3200 rows, a simple loop or list comprehension is fine
                for r in results:
                    if r["prompt"] == prompt_name and r["idx"] == idx:
                        already_done = True
                        prompt_cost += r.get("cost_usd", 0)
                        if r.get("predicted_label") == "unknown":
                            unknowns += 1
                        break
            
            if already_done:
                continue

            result = call_fn(row["premise"], row["hypothesis"], prompt_template)

            result_row = {
                "idx": idx,
                "prompt": prompt_name,
                "model": model_name,
                "set": "matched",
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

            if "reasoning_tokens" in result:
                result_row["reasoning_tokens"] = result["reasoning_tokens"]

            results.append(result_row)
            prompt_cost += result["cost_usd"]
            if result["predicted_label"] == "unknown":
                unknowns += 1

            # Save checkpoint every 20 API calls to be safe
            if len(results) % 20 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)

            # Rate limiting
            time.sleep(1.0)

        total_cost += prompt_cost
        print(f"  {prompt_name}: cost=${prompt_cost:.4f}, unknowns={unknowns}")

    # Save final results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path} ({len(df_results)} rows)")
    print(f"   Total cost: ${total_cost:.4f}")

    return df_results


def evaluate_model_results(df_results, model_name):
    """Print evaluation metrics for a model."""
    labels = ["entailment", "neutral", "contradiction"]

    print(f"\n{'#'*60}")
    print(f"# {model_name} EVALUATION")
    print(f"{'#'*60}")

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
        total_cost = subset["cost_usd"].sum()

        print(f"\n--- {prompt_name} ---")
        print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Macro F1:    {f1:.4f}")
        for i, label in enumerate(labels):
            print(f"  {label:15s} F1: {per_class[i]:.4f}")
        print(f"  Unknowns:    {unknowns}")
        print(f"  Total cost:  ${total_cost:.4f}")


def main():
    df_matched = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    print(f"Matched test set: {len(df_matched)} samples")

    # GPT-5 (o3-mini)
    print("\n" + "="*60 + "\nMODEL 1: GPT-5 (o3-mini)\n" + "="*60)
    gpt5_path = os.path.join(RESULTS_DIR, "api_results_gpt5.csv")
    try:
        df_gpt5 = run_model(df_matched, call_gpt5, "gpt5_o3mini", gpt5_path)
        evaluate_model_results(df_gpt5, "GPT-5 (o3-mini)")
    except Exception as e:
        print(f"⚠️ GPT-5 failed: {e}")

    # Claude Sonnet 4.5
    print("\n" + "="*60 + "\nMODEL 2: Claude Sonnet 4.5\n" + "="*60)
    claude_path = os.path.join(RESULTS_DIR, "api_results_claude.csv")
    try:
        df_claude = run_model(df_matched, call_claude, "claude_sonnet", claude_path)
        evaluate_model_results(df_claude, "Claude Sonnet 4.5")
    except Exception as e:
        print(f"⚠️ Claude failed: {e}")

    # Llama 3.3 70B (Groq)
    print("\n" + "="*60 + "\nMODEL 3: Llama 3.3 70B (Groq)\n" + "="*60)
    llama_path = os.path.join(RESULTS_DIR, "api_results_llama.csv")
    try:
        df_llama = run_model(df_matched, call_llama_groq, "llama_3.3_70b", llama_path)
        evaluate_model_results(df_llama, "Llama 3.3 70B")
    except Exception as e:
        print(f"⚠️ Llama failed: {e}")


if __name__ == "__main__":
    main()
