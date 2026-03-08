#!/usr/bin/env python3
"""
Notebook 02 — Encoder Baselines
=================================
Runs 5 pretrained NLI encoder models on matched (800) and mismatched (400) test sets.
Saves predictions with confidence scores for use in hybrid gatekeeper.

Models:
    1. textattack/bert-base-uncased-MNLI
    2. cross-encoder/nli-deberta-v3-small
    3. textattack/roberta-base-MNLI
    4. cross-encoder/nli-deberta-v3-base
    5. cross-encoder/nli-deberta-v3-large  [NEW — SOTA ~91.8% MNLI-matched]

Outputs:
    results/encoder_predictions_matched.csv
    results/encoder_predictions_mm.csv
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (this will be slow for large models)")

# Models to evaluate
# Two label mapping schemes:
#   textattack models: 0=entailment, 1=neutral, 2=contradiction
#   cross-encoder models: 0=contradiction, 1=entailment, 2=neutral
MODELS = [
    {
        "name": "bert_base",
        "model_id": "textattack/bert-base-uncased-MNLI",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
    {
        "name": "deberta_v3_small",
        "model_id": "cross-encoder/nli-deberta-v3-small",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
    {
        "name": "roberta_base",
        "model_id": "textattack/roberta-base-MNLI",
        "label_map": {0: "contradiction", 1: "neutral", 2: "entailment"},
    },
    {
        "name": "deberta_v3_base",
        "model_id": "cross-encoder/nli-deberta-v3-base",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
    # ── NEW: DeBERTa-v3-large (SOTA encoder, ~91.8% MNLI-matched) ──────────
    # Trained on SNLI + MultiNLI. Larger model: ~350M params vs ~86M for base.
    # Runs slower (~3× per batch) but fits on MPS / any GPU with 3GB+ VRAM.
    # Reference: He et al. (ICLR 2023) — DeBERTaV3 paper.
    {
        "name": "deberta_v3_large",
        "model_id": "cross-encoder/nli-deberta-v3-large",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
]


def run_encoder(model_info, df, device, batch_size=16):
    """
    Run a single encoder model on the given DataFrame.
    Returns lists of predictions and confidence scores.
    """
    name = model_info["name"]
    model_id = model_info["model_id"]
    label_map = model_info["label_map"]

    print(f"\n{'='*60}")
    print(f"Running: {name} ({model_id})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    predictions = []
    confidences = []

    # Process in batches for efficiency
    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"{name}"):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        premises = batch["premise"].tolist()
        hypotheses = batch["hypothesis"].tolist()

        # Tokenize
        inputs = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confs, pred_ids = probs.max(dim=1)

        # Convert to labels
        for conf, pred_id in zip(confs.cpu().numpy(), pred_ids.cpu().numpy()):
            predictions.append(label_map[int(pred_id)])
            confidences.append(float(conf))

    # Clean up GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return predictions, confidences


def evaluate_encoder(df, pred_col, conf_col, label_col="label_text", set_name=""):
    """Compute and print metrics for a single encoder."""
    y_true = df[label_col]
    y_pred = df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    labels = ["entailment", "neutral", "contradiction"]
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    print(f"\n--- {pred_col} on {set_name} ---")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1:  {f1:.4f}")
    for i, label in enumerate(labels):
        print(f"  {label:15s} F1: {per_class_f1[i]:.4f}")

    # Per-genre accuracy
    print(f"\n  Per-genre accuracy:")
    for genre in sorted(df["genre"].unique()):
        mask = df["genre"] == genre
        genre_acc = accuracy_score(df.loc[mask, label_col], df.loc[mask, pred_col])
        print(f"    {genre:15s}: {genre_acc:.4f} ({genre_acc*100:.1f}%)")

    # Mean confidence
    mean_conf = df[conf_col].mean()
    print(f"\n  Mean confidence: {mean_conf:.4f}")

    return {
        "model": pred_col.replace("_pred", ""),
        "set": set_name,
        "accuracy": acc,
        "macro_f1": f1,
        "ent_f1": per_class_f1[0],
        "neu_f1": per_class_f1[1],
        "con_f1": per_class_f1[2],
        "mean_conf": mean_conf,
    }


def threshold_analysis(df, pred_col, conf_col, label_col="label_text"):
    """Analyze accuracy at different confidence thresholds for DeBERTa-v3-base."""
    print(f"\n--- Confidence Threshold Analysis ({pred_col}) ---")
    print(f"{'Threshold':>10} | {'N covered':>10} | {'Coverage %':>10} | {'Accuracy':>10}")
    print("-" * 50)

    for threshold in [0.80, 0.85, 0.90, 0.95]:
        mask = df[conf_col] >= threshold
        n_covered = mask.sum()
        coverage = n_covered / len(df) * 100
        if n_covered > 0:
            acc = accuracy_score(df.loc[mask, label_col], df.loc[mask, pred_col])
        else:
            acc = 0.0
        print(f"    ≥{threshold:.2f} | {n_covered:>10} | {coverage:>9.1f}% | {acc:>9.4f}")


def main():
    # Load test sets
    print("Loading test sets...")
    df_matched = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_mismatched = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))
    print(f"Matched test:    {len(df_matched)} samples")
    print(f"Mismatched test: {len(df_mismatched)} samples")

    # ========================================================
    # Run all encoders on MATCHED test set
    # ========================================================
    print("\n" + "#" * 60)
    print("# RUNNING ENCODERS ON MATCHED TEST SET (800)")
    print("#" * 60)

    for model_info in MODELS:
        name = model_info["name"]
        preds, confs = run_encoder(model_info, df_matched, DEVICE)
        df_matched[f"{name}_pred"] = preds
        df_matched[f"{name}_conf"] = confs

    # Save matched predictions
    matched_path = os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv")
    df_matched.to_csv(matched_path, index=False)
    print(f"\n✅ Saved: {matched_path}")

    # ========================================================
    # Run all encoders on MISMATCHED test set
    # ========================================================
    print("\n" + "#" * 60)
    print("# RUNNING ENCODERS ON MISMATCHED TEST SET (400)")
    print("#" * 60)

    for model_info in MODELS:
        name = model_info["name"]
        preds, confs = run_encoder(model_info, df_mismatched, DEVICE)
        df_mismatched[f"{name}_pred"] = preds
        df_mismatched[f"{name}_conf"] = confs

    # Save mismatched predictions
    mm_path = os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv")
    df_mismatched.to_csv(mm_path, index=False)
    print(f"\n✅ Saved: {mm_path}")

    # ========================================================
    # Evaluate all encoders
    # ========================================================
    print("\n" + "#" * 60)
    print("# EVALUATION RESULTS")
    print("#" * 60)

    all_metrics = []

    for model_info in MODELS:
        name = model_info["name"]
        pred_col = f"{name}_pred"
        conf_col = f"{name}_conf"

        # Matched
        metrics_m = evaluate_encoder(df_matched, pred_col, conf_col, set_name="MATCHED")
        all_metrics.append(metrics_m)

        # Mismatched
        metrics_mm = evaluate_encoder(df_mismatched, pred_col, conf_col, set_name="MISMATCHED")
        all_metrics.append(metrics_mm)

    # ========================================================
    # Confidence threshold analysis for DeBERTa-v3-base
    # ========================================================
    print("\n" + "#" * 60)
    print("# CONFIDENCE THRESHOLD ANALYSIS (DeBERTa-v3-base)")
    print("#" * 60)

    threshold_analysis(
        df_matched,
        "deberta_v3_base_pred",
        "deberta_v3_base_conf",
    )

    threshold_analysis(
        df_mismatched,
        "deberta_v3_base_pred",
        "deberta_v3_base_conf",
    )

    # ========================================================
    # Summary table: Matched vs Mismatched comparison
    # ========================================================
    print("\n" + "#" * 60)
    print("# MATCHED vs MISMATCHED COMPARISON")
    print("#" * 60)

    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))

    # Compute drop
    print("\n--- Accuracy Drop (Matched → Mismatched) ---")
    for model_info in MODELS:
        name = model_info["name"]
        m_row = metrics_df[(metrics_df["model"] == name) & (metrics_df["set"] == "MATCHED")]
        mm_row = metrics_df[(metrics_df["model"] == name) & (metrics_df["set"] == "MISMATCHED")]
        if len(m_row) > 0 and len(mm_row) > 0:
            drop = mm_row.iloc[0]["accuracy"] - m_row.iloc[0]["accuracy"]
            print(f"  {name:25s}: {drop:+.4f} ({drop*100:+.1f}%)")

    print("\n" + "=" * 60)
    print("ENCODER BASELINES COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
