"""
05b_hybrid_v3_deberta_gpt4o_32shot.py
======================================
Hybrid Gatekeeper v3
  Gate  : DeBERTa-v3-base-mnli (local, free)
  LLM   : GPT-4o with 32-example few-shot prompt (P3-32)
  Logic : if max_softmax_conf >= theta  → accept encoder label
          else                          → call GPT-4o 32-shot

Key design decisions
- 32 examples selected from dev set (nli_dev_200.csv), NOT test set
  to avoid contamination.  Balanced: 10–11 per class, covering all 5 genres.
- Examples chosen to be hard / boundary cases that teach the model
  the 5 most common error patterns identified in error_analysis.csv:
    Ent→Neu (lexical overlap), Neu→Ent (plausibility trap),
    Neu→Con (over-extrapolation), Con→Neu (implicit negation),
    rhetorical-question premise.
- Thresholds tested: 0.85, 0.90, 0.95 (same as v1/v2)
- Saves: results/hybrid_v3_results.csv
- Checkpoints every 50 API rows to results/hybrid_v3_checkpoint.csv
- Cost tracking per row
"""

import os, json, time, random
import pandas as pd
import numpy as np
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
random.seed(42)
np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS    = os.path.join(BASE_DIR, "results")
CKPT_PATH  = os.path.join(RESULTS, "hybrid_v3_checkpoint.csv")
OUT_PATH   = os.path.join(RESULTS, "hybrid_v3_results.csv")
os.makedirs(RESULTS, exist_ok=True)

# ── GPT-4o pricing (per token) ──────────────────────────────────────────────
GPT4O_IN  = 2.50 / 1_000_000   # $2.50 / 1M input tokens
GPT4O_OUT = 10.00 / 1_000_000  # $10.00 / 1M output tokens

# ── device ─────────────────────────────────────────────────────────────────
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"Device: {device}")

# ── 32-shot examples ────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    # ── ENTAILMENT (11) ────────────────────────────────────────────────────
    {
        "premise":    "I said it and I'm glad.",
        "hypothesis": "I'm glad I said it.",
        "label":      "entailment",
        "note":       "Idiomatic paraphrase — reordering does not change truth value."
    },
    {
        "premise":    "Drinks are available and expensive.",
        "hypothesis": "Drinks cost a lot.",
        "label":      "entailment",
        "note":       "Partial entailment: 'expensive' strictly entails 'cost a lot'."
    },
    {
        "premise":    "those little kids don't understand it",
        "hypothesis": "Those young children can't comprehend it.",
        "label":      "entailment",
        "note":       "Lexical substitution: 'little kids'='young children', 'understand'='comprehend'."
    },
    {
        "premise":    "I've always threatened to take lessons but I've never gotten around to it.",
        "hypothesis": "I have never gotten around to taking lessons.",
        "label":      "entailment",
        "note":       "Spoken register — informal phrasing directly entails formal hypothesis."
    },
    {
        "premise":    "Cybernetics had always been Derry's passion.",
        "hypothesis": "Derry had a passion for cybernetics.",
        "label":      "entailment",
        "note":       "Passive/active restructuring preserves truth."
    },
    {
        "premise":    "Duke William returned from his conquest of England to attend the consecration of Notre-Dame in 1067.",
        "hypothesis": "Duke William attended the consecration of Notre-Dame.",
        "label":      "entailment",
        "note":       "Hypothesis drops the year and reason but retains the core fact."
    },
    {
        "premise":    "they take the football serious",
        "hypothesis": "Football is important to them.",
        "label":      "entailment",
        "note":       "Colloquial 'take... serious' entails 'is important to them'."
    },
    {
        "premise":    "Most of it, I couldn't even begin to identify.",
        "hypothesis": "I didn't know what any of it was.",
        "label":      "entailment",
        "note":       "Idiomatic 'couldn't even begin to identify' = 'didn't know'. Do NOT choose neutral."
    },
    {
        "premise":    "'I see.'",
        "hypothesis": "I saw it.",
        "label":      "entailment",
        "note":       "Conversational acknowledgement 'I see' entails prior visual perception."
    },
    {
        "premise":    "I smiled vaguely.",
        "hypothesis": "I moved my face.",
        "label":      "entailment",
        "note":       "Smiling is a specific type of moving one's face — strict entailment."
    },
    {
        "premise":    "The best beach in Europe — at least that's the verdict of its regulars.",
        "hypothesis": "Regular beachgoers say that it is the best in Europe.",
        "label":      "entailment",
        "note":       "'verdict of its regulars' = 'regular beachgoers say'."
    },

    # ── NEUTRAL (11) ───────────────────────────────────────────────────────
    {
        "premise":    "That's the second time you've made that sort of remark.",
        "hypothesis": "That's the second time you've made that sort of remark and I don't need any more reminders.",
        "label":      "neutral",
        "note":       "Hypothesis adds new content ('I don't need reminders') not in premise."
    },
    {
        "premise":    "Restored in 1967, the beautiful exterior is complemented by the fine period furniture housed inside.",
        "hypothesis": "The restorations done to the structure are very promising.",
        "label":      "neutral",
        "note":       "Premise describes completed restoration; hypothesis adds an evaluative claim not stated."
    },
    {
        "premise":    "oh i don't know either the other growing up all i knew was",
        "hypothesis": "When I was 7 the only thing I knew was",
        "label":      "neutral",
        "note":       "Hypothesis adds specific age '7' not mentioned in premise — plausible but not entailed."
    },
    {
        "premise":    "From there, take the road that heads back to the coast and Es Pujols, Formentera's premier resort village.",
        "hypothesis": "Formentera's premier resort village has swimming pools, bars and restaurants.",
        "label":      "neutral",
        "note":       "Resort village is mentioned but no amenities are stated — hypothesis is possible, not certain."
    },
    {
        "premise":    "yeah pay fifteen yeah yes i know yeah and when you pay fifteen dollars a month it sure takes a long time",
        "hypothesis": "When you pay $15 a month, it takes a long time but I can't afford any more.",
        "label":      "neutral",
        "note":       "Hypothesis adds 'I can't afford any more' which is not in the premise."
    },
    {
        "premise":    "This tourist heartland is also home to 100,000 Jamaicans who live in the hills surrounding the town.",
        "hypothesis": "Beautiful white beaches are the reason this town is so popular with tourists.",
        "label":      "neutral",
        "note":       "Premise confirms tourists visit; hypothesis gives a cause not stated in premise."
    },
    {
        "premise":    "A funny place for a piece of brown paper, I mused.",
        "hypothesis": "I looked down at my desk, which was a mess, as usual, and had some white and brown papers on it.",
        "label":      "neutral",
        "note":       "Brown paper is mentioned but desk, mess, and white papers are all additions."
    },
    {
        "premise":    "Why bother to sacrifice your lives for dirt farmers and slavers?",
        "hypothesis": "People sacrifice their lives for farmers and slaves.",
        "label":      "neutral",
        "note":       "CRITICAL: Premise is a RHETORICAL QUESTION — it does NOT assert that people sacrifice. Choose neutral, not entailment."
    },
    {
        "premise":    "be of good cheer,",
        "hypothesis": "Be of good cheer, for beer time is near.",
        "label":      "neutral",
        "note":       "Hypothesis adds new content (beer time) not entailed by premise."
    },
    {
        "premise":    "Traffic has been controlled, and if you're staying here you might want to consider getting around by bicycle.",
        "hypothesis": "It is quicker to cross the island by bike than by car.",
        "label":      "neutral",
        "note":       "Cycling is recommended but speed comparison is not stated."
    },
    {
        "premise":    "yeah well the jury that originally sentenced him sentenced him to death",
        "hypothesis": "The sentence later got revised under review of a judge.",
        "label":      "neutral",
        "note":       "Premise states original sentence; hypothesis adds a future revision not stated."
    },

    # ── CONTRADICTION (10) ─────────────────────────────────────────────────
    {
        "premise":    "and uh it that takes so much time away from your kids",
        "hypothesis": "Leaves you with plenty of time for your kids.",
        "label":      "contradiction",
        "note":       "Takes time AWAY vs. leaves you WITH time — direct opposition."
    },
    {
        "premise":    "yeah okay you go ahead",
        "hypothesis": "No, do not go ahead.",
        "label":      "contradiction",
        "note":       "Direct lexical opposition."
    },
    {
        "premise":    "Each of them was as tough as a thick tree and loyal to the death.",
        "hypothesis": "None of them were loyal to anything.",
        "label":      "contradiction",
        "note":       "'loyal to the death' directly contradicts 'not loyal to anything'."
    },
    {
        "premise":    "She had thrown away her cloak and tied her hair back into a topknot to keep it out of the way.",
        "hypothesis": "She shaved her head.",
        "label":      "contradiction",
        "note":       "Tying hair into a topknot implies she has hair; shaving implies she doesn't."
    },
    {
        "premise":    "To check this, the central bank has tripled interest rates.",
        "hypothesis": "The bank doubled interest rates as a way of checking this.",
        "label":      "contradiction",
        "note":       "Tripled ≠ doubled — precise numeric contradiction."
    },
    {
        "premise":    "excessively violent — I was worried it's like golly if kids start imitating that",
        "hypothesis": "It was non-violent, so it would be great for the kids to follow their lead.",
        "label":      "contradiction",
        "note":       "'excessively violent' directly contradicts 'non-violent'."
    },
    {
        "premise":    "I don't know what I would have done without Legal Services, said James.",
        "hypothesis": "James said Legal Services was of no help.",
        "label":      "contradiction",
        "note":       "Implicit negation: praising a service contradicts saying it was 'no help'."
    },
    {
        "premise":    "The cane plantations, increasingly in the hands of American tycoons, found a ready market in the US.",
        "hypothesis": "The US market was not ready for the cane plantations.",
        "label":      "contradiction",
        "note":       "'found a ready market' directly contradicts 'market was not ready'."
    },
    {
        "premise":    "um-hum with the ice yeah",
        "hypothesis": "With the sunshine and heat wave yes.",
        "label":      "contradiction",
        "note":       "Ice (cold) vs heat wave — environmental contradiction."
    },
    {
        "premise":    "but how do you know the good from the bad",
        "hypothesis": "Why care if it's good or bad?",
        "label":      "contradiction",
        "note":       "Premise implies distinguishing good/bad matters; hypothesis says it doesn't."
    },
]

assert len(FEW_SHOT_EXAMPLES) == 32, f"Expected 32 examples, got {len(FEW_SHOT_EXAMPLES)}"


def build_system_prompt() -> str:
    """Build the GPT-4o system prompt with 32 few-shot examples."""
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += (
            f"\nExample {i}:\n"
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"Label: {ex['label']}\n"
        )

    return f"""You are an expert Natural Language Inference classifier.

TASK: Given a Premise and Hypothesis, output exactly one label:
  entailment   — the hypothesis is necessarily TRUE if the premise is true
  contradiction — the hypothesis is necessarily FALSE if the premise is true
  neutral      — the hypothesis is neither necessarily true nor necessarily false

CRITICAL RULES (these override intuition):
1. ENTAILMENT requires STRICT logical necessity, not just plausibility.
   If the hypothesis COULD be true but is not GUARANTEED by the premise → neutral.
2. RHETORICAL QUESTIONS: If the premise is a question (even rhetorical),
   it makes NO assertion. A hypothesis asserting what the question implies → neutral.
3. IDIOMATIC EQUIVALENCE: Colloquial phrases like "couldn't even begin to X"
   = "didn't X at all". Map idioms to their literal equivalents before labelling.
4. IMPLICIT NEGATION: Watch for praise/criticism reversals. Saying a service
   "saved me" contradicts "was no help" even without explicit negation.
5. NUMERIC PRECISION: "tripled" ≠ "doubled". Precise numbers matter.
6. ADDED CONTENT: If the hypothesis adds ANY new claim not in the premise → neutral
   (unless the addition directly contradicts the premise → contradiction).

OUTPUT FORMAT: Respond with exactly one word: entailment, contradiction, or neutral.
Do not include punctuation, explanation, or any other text.

EXAMPLES (study these carefully — they illustrate the rules above):
{examples_text}
Now classify the following:"""


def parse_label(raw: str) -> str:
    raw = raw.strip().lower().rstrip(".")
    if raw in ("entailment", "contradiction", "neutral"):
        return raw
    for label in ("entailment", "contradiction", "neutral"):
        if label in raw:
            return label
    return "neutral"  # safe fallback


def call_gpt4o(client: OpenAI, system: str, premise: str, hypothesis: str,
               retries: int = 3) -> tuple:
    """Returns (predicted_label, prompt_tokens, completion_tokens, cost_usd)."""
    user_msg = f"Premise: {premise}\nHypothesis: {hypothesis}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
                seed=42,
            )
            raw   = resp.choices[0].message.content
            pt    = resp.usage.prompt_tokens
            ct    = resp.usage.completion_tokens
            cost  = pt * GPT4O_IN + ct * GPT4O_OUT
            return parse_label(raw), pt, ct, cost
        except Exception as e:
            wait = 2 ** attempt
            print(f"  API error ({e}), retry in {wait}s...")
            time.sleep(wait)
    return "neutral", 0, 0, 0.0


def run_hybrid_v3(df, encoder_preds, threshold, client, system_prompt,
                  set_name, ckpt_path):

    # Load checkpoint
    done_idx = set()
    rows = []
    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path)
        ckpt_t = ckpt[(ckpt["threshold"] == threshold) & (ckpt["set"] == set_name)]
        done_idx = set(ckpt_t["idx"].tolist())
        rows = ckpt_t.to_dict("records")
        print(f"  Resuming from checkpoint: {len(done_idx)} done")

    api_calls = 0
    for i, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"v3 θ={threshold} [{set_name}]"):
        if i in done_idx:
            continue

        enc_row  = encoder_preds.iloc[i]
        enc_pred = enc_row["deberta_v3_base_pred"]
        enc_conf = enc_row["deberta_v3_base_conf"]

        if enc_conf >= threshold:
            rows.append({
                "idx": i, "hybrid": "v3_deberta_gpt4o_32shot",
                "threshold": threshold, "set": set_name,
                "premise": row["premise"], "hypothesis": row["hypothesis"],
                "genre": row.get("genre", ""),
                "label_true": row["label_text"],
                "label_pred": enc_pred,
                "source": "encoder",
                "enc_conf": enc_conf,
                "tokens": 0, "cost_usd": 0.0,
            })
        else:
            pred, pt, ct, cost = call_gpt4o(
                client, system_prompt, row["premise"], row["hypothesis"]
            )
            api_calls += 1
            rows.append({
                "idx": i, "hybrid": "v3_deberta_gpt4o_32shot",
                "threshold": threshold, "set": set_name,
                "premise": row["premise"], "hypothesis": row["hypothesis"],
                "genre": row.get("genre", ""),
                "label_true": row["label_text"],
                "label_pred": pred,
                "source": "api",
                "enc_conf": enc_conf,
                "tokens": pt + ct, "cost_usd": cost,
            })

            if api_calls % 50 == 0:
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                print(f"  Checkpoint saved ({api_calls} API calls)")

    return pd.DataFrame(rows)


def evaluate(df):
    correct = (df["label_true"] == df["label_pred"]).sum()
    total   = len(df)
    acc     = correct / total
    enc_pct = (df["source"] == "encoder").mean() * 100
    api_pct = (df["source"] == "api").mean() * 100
    cost_1k = df["cost_usd"].sum() / total * 1000
    return {
        "accuracy":  round(acc * 100, 2),
        "encoder_%": round(enc_pct, 1),
        "api_%":     round(api_pct, 1),
        "cost_1k":   round(cost_1k, 4),
        "n":         total,
    }


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_m  = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))
    enc_m   = pd.read_csv(os.path.join(RESULTS, "encoder_predictions_matched.csv"))
    enc_mm  = pd.read_csv(os.path.join(RESULTS, "encoder_predictions_mm.csv"))

    system_prompt = build_system_prompt()
    print(f"System prompt length: ~{len(system_prompt)//4} tokens")

    thresholds = [0.85, 0.90, 0.95]
    all_results = []

    for theta in thresholds:
        print(f"\n{'='*60}")
        print(f"Running Hybrid v3  θ={theta}")
        print(f"{'='*60}")

        ckpt_m = CKPT_PATH.replace(".csv", f"_m_{theta}.csv")
        df_m = run_hybrid_v3(test_m, enc_m, theta, client,
                             system_prompt, "matched", ckpt_m)
        stats_m = evaluate(df_m)
        print(f"  Matched   → Acc={stats_m['accuracy']}%  "
              f"API={stats_m['api_%']}%  Cost=${stats_m['cost_1k']}/1k")

        ckpt_mm = CKPT_PATH.replace(".csv", f"_mm_{theta}.csv")
        df_mm = run_hybrid_v3(test_mm, enc_mm, theta, client,
                              system_prompt, "mismatched", ckpt_mm)
        stats_mm = evaluate(df_mm)
        print(f"  Mismatched → Acc={stats_mm['accuracy']}%  "
              f"API={stats_mm['api_%']}%  Cost=${stats_mm['cost_1k']}/1k")

        combined = pd.concat([df_m, df_mm], ignore_index=True)
        all_results.append(combined)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")

    print("\n" + "="*70)
    print("HYBRID V3 RESULTS — DeBERTa-mnli + GPT-4o 32-shot")
    print("="*70)
    print(f"{'θ':<6} {'Set':<12} {'Acc':>7} {'Enc%':>7} {'API%':>7} {'Cost/1k':>10}")
    print("-"*70)
    for theta in thresholds:
        for set_name in ["matched", "mismatched"]:
            subset = final[(final["threshold"] == theta) & (final["set"] == set_name)]
            if len(subset) == 0:
                continue
            s = evaluate(subset)
            print(f"{theta:<6} {set_name:<12} {s['accuracy']:>6}%"
                  f" {s['encoder_%']:>6}%  {s['api_%']:>5}%"
                  f"  ${s['cost_1k']:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
