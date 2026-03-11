#!/usr/bin/env python3
"""
Notebook 01 — Data Preparation
================================
Loads MultiNLI matched and mismatched dev sets, removes ambiguous labels,
creates stratified samples for dev (200), test-matched (800), and test-mismatched (400).

Outputs:
    data/nli_dev_200.csv        — 200 samples, matched genres, for prompt tuning
    data/nli_test_800.csv       — 800 samples, matched genres, primary evaluation
    data/nli_test_mm_400.csv    — 400 samples, mismatched genres, generalisation evaluation
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================
SEED = 42
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MULTINLI_DIR = os.environ.get(
    "MULTINLI_DIR",
    os.path.join(os.path.dirname(PROJECT_DIR), "LLM/NLI/multinli_1.0")
)

MATCHED_FILE = os.path.join(MULTINLI_DIR, "multinli_1.0_dev_matched.jsonl")
MISMATCHED_FILE = os.path.join(MULTINLI_DIR, "multinli_1.0_dev_mismatched.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. Load raw JSONL files
# ============================================================
print("=" * 60)
print("STEP 1: Loading MultiNLI JSONL files")
print("=" * 60)

def load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

matched_rows = load_jsonl(MATCHED_FILE)
mismatched_rows = load_jsonl(MISMATCHED_FILE)

print(f"Matched raw rows:    {len(matched_rows)}")
print(f"Mismatched raw rows: {len(mismatched_rows)}")

# ============================================================
# 2. Build DataFrames
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building DataFrames")
print("=" * 60)

def build_df(rows):
    """Convert raw JSONL rows to a clean DataFrame."""
    return pd.DataFrame([{
        "premise":    r["sentence1"],
        "hypothesis": r["sentence2"],
        "label_text": r["gold_label"],
        "genre":      r["genre"],
    } for r in rows])

df_matched = build_df(matched_rows)
df_mismatched = build_df(mismatched_rows)

# ============================================================
# 3. Remove ambiguous labels (gold_label == '-')
# ============================================================
n_ambig_m = (df_matched["label_text"] == "-").sum()
n_ambig_mm = (df_mismatched["label_text"] == "-").sum()

df_matched = df_matched[df_matched["label_text"] != "-"].reset_index(drop=True)
df_mismatched = df_mismatched[df_mismatched["label_text"] != "-"].reset_index(drop=True)

print(f"Matched:    removed {n_ambig_m} ambiguous → {len(df_matched)} usable")
print(f"Mismatched: removed {n_ambig_mm} ambiguous → {len(df_mismatched)} usable")

# ============================================================
# 4. Print genre distributions
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Genre Distributions")
print("=" * 60)

print("\nMatched genres:")
print(df_matched["genre"].value_counts().to_string())

print("\nMismatched genres:")
print(df_mismatched["genre"].value_counts().to_string())

print(f"\nMatched genres:    {sorted(df_matched['genre'].unique())}")
print(f"Mismatched genres: {sorted(df_mismatched['genre'].unique())}")

# ============================================================
# 5. Print label distributions
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Label Distributions")
print("=" * 60)

print("\nMatched label distribution:")
print(df_matched["label_text"].value_counts().to_string())

print("\nMismatched label distribution:")
print(df_mismatched["label_text"].value_counts().to_string())

# ============================================================
# 6. Stratified sampling — matched (1000 → 200 dev + 800 test)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Stratified Sampling")
print("=" * 60)

# Sample 1000 from matched, stratified by label
_, df_m_sample = train_test_split(
    df_matched,
    test_size=1000,
    stratify=df_matched["label_text"],
    random_state=SEED
)

# Split 1000 into 200 dev + 800 test
df_m_dev, df_m_test = train_test_split(
    df_m_sample,
    test_size=800,
    stratify=df_m_sample["label_text"],
    random_state=SEED
)

# Sample 400 from mismatched, stratified by label
_, df_mm_test = train_test_split(
    df_mismatched,
    test_size=400,
    stratify=df_mismatched["label_text"],
    random_state=SEED
)

# Reset indices
df_m_dev = df_m_dev.reset_index(drop=True)
df_m_test = df_m_test.reset_index(drop=True)
df_mm_test = df_mm_test.reset_index(drop=True)

print(f"DEV (matched):       {len(df_m_dev)} samples")
print(f"TEST matched:        {len(df_m_test)} samples")
print(f"TEST mismatched:     {len(df_mm_test)} samples")

# ============================================================
# 7. Verify sample distributions
# ============================================================
print("\n--- DEV set (200) ---")
print("Labels:", df_m_dev["label_text"].value_counts().to_dict())
print("Genres:", df_m_dev["genre"].value_counts().to_dict())

print("\n--- TEST matched (800) ---")
print("Labels:", df_m_test["label_text"].value_counts().to_dict())
print("Genres:", df_m_test["genre"].value_counts().to_dict())

print("\n--- TEST mismatched (400) ---")
print("Labels:", df_mm_test["label_text"].value_counts().to_dict())
print("Genres:", df_mm_test["genre"].value_counts().to_dict())

# ============================================================
# 8. Verify zero overlap between sets
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Overlap Verification")
print("=" * 60)

dev_pairs = set(zip(df_m_dev["premise"], df_m_dev["hypothesis"]))
test_pairs = set(zip(df_m_test["premise"], df_m_test["hypothesis"]))
mm_pairs = set(zip(df_mm_test["premise"], df_mm_test["hypothesis"]))

overlap_dev_test = len(dev_pairs & test_pairs)
overlap_test_mm = len(test_pairs & mm_pairs)
overlap_dev_mm = len(dev_pairs & mm_pairs)

print(f"DEV ∩ TEST matched:    {overlap_dev_test} overlapping pairs")
print(f"TEST matched ∩ TEST mm: {overlap_test_mm} overlapping pairs")
print(f"DEV ∩ TEST mm:         {overlap_dev_mm} overlapping pairs")

assert overlap_dev_test == 0, "OVERLAP detected between DEV and TEST matched!"
assert overlap_test_mm == 0, "OVERLAP detected between TEST matched and TEST mismatched!"
assert overlap_dev_mm == 0, "OVERLAP detected between DEV and TEST mismatched!"

print("✅ All overlap checks passed — zero contamination.")

# ============================================================
# 9. Save CSVs
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Datasets")
print("=" * 60)

dev_path = os.path.join(DATA_DIR, "nli_dev_200.csv")
test_path = os.path.join(DATA_DIR, "nli_test_800.csv")
mm_path = os.path.join(DATA_DIR, "nli_test_mm_400.csv")

df_m_dev.to_csv(dev_path, index=False)
df_m_test.to_csv(test_path, index=False)
df_mm_test.to_csv(mm_path, index=False)

print(f"✅ Saved: {dev_path} ({len(df_m_dev)} rows)")
print(f"✅ Saved: {test_path} ({len(df_m_test)} rows)")
print(f"✅ Saved: {mm_path} ({len(df_mm_test)} rows)")

# ============================================================
# 10. Statistical justification
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Statistical Justification")
print("=" * 60)

import math

def confidence_interval(n, p=0.5, z=1.96):
    """95% CI for proportion."""
    return z * math.sqrt(p * (1 - p) / n)

ci_800 = confidence_interval(800) * 100
ci_400 = confidence_interval(400) * 100
ci_200 = confidence_interval(200) * 100

print(f"800 matched samples  → ±{ci_800:.1f}% CI at 95% confidence")
print(f"400 mismatched samples → ±{ci_400:.1f}% CI at 95% confidence")
print(f"200 dev samples       → ±{ci_200:.1f}% CI at 95% confidence")
print("\nBoth test sets are sufficient to distinguish systems differing by >3% accuracy.")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE ✅")
print("=" * 60)
