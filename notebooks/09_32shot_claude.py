#!/usr/bin/env python3
"""
Notebook 09 — 32-Shot Claude Sonnet Evaluation
================================================
Evaluates Claude Sonnet with a curated 32-shot prompt on the matched test set.

The 32 examples are handpicked from the dev set (nli_dev_200.csv), never overlapping
with the test set. They were selected to maximally cover:
  - All 5 genres (fiction, government, slate, telephone, travel) — ~6-7 examples each
  - All 3 labels (entailment, neutral, contradiction) — balanced ~10-11 each
  - Hard linguistic patterns:
      * Rhetorical questions (neutral mislabelled as entailment)
      * Lexical overlap traps (neutral mislabelled as contradiction)
      * Colloquialisms and idioms (entailment mislabelled as neutral)
      * Negation (contradiction mislabelled as neutral)
      * Implicit inference (entailment requiring world knowledge)
      * Out-of-context fragments (telephone transcripts)

Output:
    results/api_results_claude_32shot.csv
"""

import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 32-Shot Examples — curated from nli_dev_200.csv
# Ordered: easy → hard, diverse genres interleaved
# ============================================================
EXAMPLES_32 = [
    # ── CLEAR / EASY EXAMPLES (anchors for the model) ──────────────────────
    # [travel|entailment] — direct factual restatement
    ("Try a selection at the Whisky Heritage Centre (they have over 100 for you to sample).",
     "There are at least 100 things to sample at Whisky Heritage Centre.",
     "entailment"),
    # [slate|entailment] — synonym mapping
    ("Until the late '60s, the Senate was deferential to the (many fewer) presidential nominees.",
     "The Senate was respectful of the presidential nominees.",
     "entailment"),
    # [government|entailment] — numeric paraphrase
    ("INTEREST RATE - The price charged per unit of money borrowed per year, or other unit of time, usually expressed as a percentage.",
     "Interest is almost always expressed in terms of percent.",
     "entailment"),
    # [telephone|entailment] — casual paraphrase
    ("you know like CODA comes out of your out of your pay and the credit union comes out of your pay so we don't have to do anything there and the rest of it as far as my salary goes i just have it automatically deposited in into our bank",
     "I set things up so that my salary automatically deposits into our bank.",
     "entailment"),
    # [fiction|entailment] — clear implication
    ("After the execution of Guru Tegh Bahadur, his son, Guru Gobind Singh, exalted the faithful to be ever ready for armed defense.",
     "Guru Tegh Bahadur has a son named Guru Gobind Singh",
     "entailment"),
    # [government|contradiction] — direct factual opposite
    ("SSA is also seeking statutory authority for additional tools to recover current overpayments.",
     "SSA wants the authority to recover underpayments.",
     "contradiction"),
    # [travel|contradiction] — specific number/location mismatch
    ("The 37 hectares (91 acres) of garden are set on lands above the Wag Wag River.",
     "The nine hundred acres of garden are set on the lands above the Nile River.",
     "contradiction"),
    # [slate|contradiction] — semantic inversion
    ("She has exchanged a hollow life for a heightened life, and has tried to comprehend all its turns, get its possibilities.",
     "She has chose to live a hollow life.",
     "contradiction"),
    # [telephone|contradiction] — clear denial
    ("oh thank God i've never been to Midland",
     "I go to Midland every other weekend.",
     "contradiction"),
    # [fiction|contradiction] — state reversal
    ("Then he sobered.",
     "He was drunk.",
     "contradiction"),
    # [slate|neutral] — genre gap (implies but does not confirm)
    ("In our family we have two sons in public life.",
     "Having Two sons in public puts strains on our family's privacy.",
     "neutral"),
    # [travel|neutral] — open-ended, cannot be determined
    ("The Romans built roads and established towns, including the towns of Palmaria (Palma) and Pollentia.",
     "Some of the original towns built by the Romans are still in existence.",
     "neutral"),
    # [government|neutral] — statement vs prediction
    ("These adaptations are not uniformly valued.",
     "The values always change",
     "neutral"),
    # [telephone|neutral] — pragmatic uncertainty
    ("yeah uh-huh but we look at it sort of as an investment in the future too",
     "The results will not be noticeable until further down the line.",
     "neutral"),
    # [fiction|neutral] — character assumption
    ("I'm not sentimental, you know.\", She paused.",
     "Everyone thinks she's sentimental.",
     "neutral"),

    # ── HARD EXAMPLES — patterns that fool encoders ─────────────────────────
    # [telephone|neutral→entailment trap] rhetorical question
    ("do you think most states have that or",
     "In your opinion, do most states have that?",
     "entailment"),  # premise IS a question → hypothesis is a restatement, so entailment
    # [telephone|entailment] colloquial→standard mapping
    ("but you know they kids seem like when they get ten or twelve years old they fall out of that",
     "Many kids leave the Scouts when they are pre-teens.",
     "entailment"),
    # [government|entailment] — indirect inference
    ("It is truly an honour.",
     "They were humbled.",
     "entailment"),
    # [slate|entailment] — indirect implication
    ("Arafat is also ailing and has no clear successor.",
     "Arafat is in bad health and does not have a person chosen to take his place.",
     "entailment"),
    # [travel|entailment] — proper noun + synonym
    ("Troyes is also a center for shopping, with two outlet centers selling both French and international designer-name fashions and home accessories.",
     "Troues had two outlet centers which sell clothes and home accessories.",
     "entailment"),
    # [government|contradiction] — near-synonym swap changes meaning
    ("Several security managers said that by participating in our study, they hoped to gain insights on how to improve their information security programs.",
     "The security managers in the study joined in order to see what we were doing wrong.",
     "contradiction"),
    # [telephone|contradiction] — lexical antonym
    ("well his knees were bothering him yeah",
     "He was in tip-top condition.",
     "contradiction"),
    # [travel|contradiction] — implicit geographic fact
    ("Jerusalem was divided into east and west, under the control of Jordan and Israel respectively.",
     "Israel won the war and Jerusalem.",
     "contradiction"),
    # [slate|contradiction] — negation flip
    ("How to Watch Washington Week in Review : Back to front.",
     "The only way to watch Washington Week in Review is from the start to the end, as anything else would be viewed.",
     "contradiction"),
    # [government|neutral] — adjacent vs required
    ("Although, in this case the equipment did not have to be erected adjacent to an operating boiler, the erection included demolishing and erecting a complete boiler island.",
     "Although it was unnecessary, some of the equipment was adjacent.",
     "neutral"),
    # [government|neutral] — long compound sentence, cannot determine
    ("The governing statute provides that a committee consisting of the Comptroller General, the Speaker of the House and President Pro Tempore of the Senate recommend an individual to the President.",
     "The process is long and will be reformed in the coming years.",
     "neutral"),
    # [travel|neutral] — implied consequence, not stated
    ("Cave 31 tries to emulate the style of the great Hindu temple on a much smaller scale, but the artists here were working on much harder rock and so abandoned their effort.",
     "Cave 31 ran into problems because it was made of harder rock and everyone was disappointed.",
     "neutral"),
    # [slate|neutral] — speculative interpretation
    ("So let me draw a slightly different moral from the saga of beach volleyball. If, as Speaker Gingrich says, the price of volleyball is eternal freedom, still it may take a village to raise a volleyball net.",
     "Speaker Gingrich thinks there is a linear connection between volleyball and freedom.",
     "neutral"),
    # [fiction|neutral] — attribution context
    ("San'doro didn't make it sound hypothetical, thought Jon.",
     "San'doro's words were hollow, and Jon knew the truth of that immediately.",
     "neutral"),
    # [telephone|neutral] — possible but not confirmed
    ("no never heard of it",
     "He does not know what it is.",
     "neutral"),  # tricky: "He" refers to self vs third party — ambiguous
    # [travel|neutral] — value judgment vs fact
    ("Many Greeks in Asia Minor were forced to leave their homes and brought an influence of eastern cadences with them.",
     "The poor Greeks shouldn't have had to leave their homes.",
     "neutral"),
    # [government|contradiction] — environmental agency swapped
    ("ENVIRONMENTAL PROTECTION AGENCY",
     "Agency which is responsible for the destruction of the environment.",
     "contradiction"),
]

assert len(EXAMPLES_32) == 32, f"Expected 32 examples, got {len(EXAMPLES_32)}"
print(f"✅ 32 examples loaded. Label distribution: "
      f"ent={sum(1 for e in EXAMPLES_32 if e[2]=='entailment')}, "
      f"neu={sum(1 for e in EXAMPLES_32 if e[2]=='neutral')}, "
      f"con={sum(1 for e in EXAMPLES_32 if e[2]=='contradiction')}")


# ============================================================
# Build the prompt
# ============================================================
def build_32shot_header():
    lines = ["Classify the logical relationship between each premise and hypothesis.\n"]
    lines.append("Labels: entailment | neutral | contradiction\n")
    lines.append("Rules:")
    lines.append("- entailment: if the premise is true, the hypothesis MUST be true")
    lines.append("- contradiction: if the premise is true, the hypothesis MUST be false")
    lines.append("- neutral: the truth of the hypothesis cannot be determined from the premise alone")
    lines.append("- Be careful with rhetorical questions, colloquialisms, and partial quotes")
    lines.append("- Respond with ONLY the label word on a single line.\n")
    lines.append("Examples:")
    lines.append("----------")
    for i, (p, h, label) in enumerate(EXAMPLES_32, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Premise: {p}")
        lines.append(f"Hypothesis: {h}")
        lines.append(f"Label: {label}")
    lines.append("\n----------")
    return "\n".join(lines)


PROMPT_HEADER = build_32shot_header()


def build_prompt(premise, hypothesis):
    return (
        PROMPT_HEADER
        + f"\n\nNow classify:\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    )


# ============================================================
# Claude client
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    first_line = text.strip().split("\n")[0].strip().lower()
    first_line = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line.startswith(label):
            return label
    text_clean = text.lower()
    for label in ["contradiction", "entailment", "neutral"]:
        if label in text_clean:
            return label
    return "unknown"


def call_claude_32shot(premise, hypothesis, max_retries=3):
    import anthropic
    client = anthropic.Anthropic(timeout=20.0)

    prompt = build_prompt(premise, hypothesis)

    INPUT_COST = 3.00
    OUTPUT_COST = 15.00

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=20,   # label only — no CoT needed; 32 examples do the work
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            label = parse_label(raw)
            input_tok = response.usage.input_tokens
            output_tok = response.usage.output_tokens
            cost = (input_tok * INPUT_COST + output_tok * OUTPUT_COST) / 1_000_000
            return {
                "raw_response": raw,
                "predicted_label": label,
                "prompt_tokens": input_tok,
                "completion_tokens": output_tok,
                "cost_usd": cost,
            }
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Claude error (attempt {attempt+1}): {e}")
            time.sleep(wait)

    return {
        "raw_response": "ERROR",
        "predicted_label": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }


# ============================================================
# Main
# ============================================================
def main():
    df_test = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    print(f"Test set: {len(df_test)} samples")

    out_path = os.path.join(RESULTS_DIR, "api_results_claude_32shot.csv")

    # Resume logic
    existing = None
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        print(f"  Resuming from {len(existing)} rows")

    results = []
    total_cost = 0.0
    unknowns = 0

    for idx in tqdm(range(len(df_test)), desc="Claude 32-shot"):
        row = df_test.iloc[idx]

        # Resume
        if existing is not None:
            mask = existing["idx"] == idx
            if mask.any():
                r = existing[mask].iloc[0].to_dict()
                results.append(r)
                total_cost += r.get("cost_usd", 0)
                if r.get("predicted_label") == "unknown":
                    unknowns += 1
                continue

        res = call_claude_32shot(row["premise"], row["hypothesis"])

        result_row = {
            "idx": idx,
            "prompt": "P5_32shot",
            "model": "claude_sonnet_32shot",
            "set": "matched",
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "genre": row["genre"],
            "label_true": row["label_text"],
            "predicted_label": res["predicted_label"],
            "raw_response": res["raw_response"],
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "cost_usd": res["cost_usd"],
        }
        results.append(result_row)
        total_cost += res["cost_usd"]
        if res["predicted_label"] == "unknown":
            unknowns += 1

        if (idx + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

        time.sleep(0.1)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path} ({len(df_out)} rows)")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Unknowns: {unknowns}")

    # Evaluate
    labels = ["entailment", "neutral", "contradiction"]
    y_true = df_out["label_true"]
    y_pred = df_out["predicted_label"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    per_class = f1_score(y_true, y_pred, average=None, labels=labels)

    print(f"\n{'='*50}")
    print(f"CLAUDE 32-SHOT RESULTS (Matched Test Set)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Macro F1:  {f1:.4f}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl:15s} F1: {per_class[i]:.4f}")
    print(f"  Cost/1k:   ${total_cost/len(df_out)*1000:.4f}")


if __name__ == "__main__":
    main()
