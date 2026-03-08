#!/usr/bin/env python3
"""
Notebook 08 — Error Analysis
==============================
Computes error type distributions and extracts real failure cases
for GPT-4o P4, Claude Sonnet P4, DeBERTa, and Hybrid v2.

Outputs:
    results/error_analysis.csv
    (Prints error cases to terminal for inclusion in the report)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

def safe_load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def analyze_errors(df_true, df_pred, model_name):
    """Compute error type distribution."""
    if df_pred is None or len(df_true) != len(df_pred):
        return None
        
    errors = df_true != df_pred
    total_errors = errors.sum()
    error_rate = total_errors / len(df_true) * 100
    
    # Types
    e_to_n = ((df_true == "entailment") & (df_pred == "neutral")).sum()
    e_to_c = ((df_true == "entailment") & (df_pred == "contradiction")).sum()
    n_to_e = ((df_true == "neutral") & (df_pred == "entailment")).sum()
    n_to_c = ((df_true == "neutral") & (df_pred == "contradiction")).sum()
    c_to_n = ((df_true == "contradiction") & (df_pred == "neutral")).sum()
    c_to_e = ((df_true == "contradiction") & (df_pred == "entailment")).sum()
    
    return {
        "model": model_name,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "Ent_to_Neu": e_to_n,
        "Ent_to_Con": e_to_c,
        "Neu_to_Ent": n_to_e,
        "Neu_to_Con": n_to_c,
        "Con_to_Neu": c_to_n,
        "Con_to_Ent": c_to_e
    }

def genre_error_breakdown(df_v2, df_enc_m):
    """Per-genre error rate for DeBERTa vs Hybrid v2."""
    # Use 0.90 threshold for the comparison as it's the standard reporting point
    sub = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
    if sub.empty:
        return
        
    print("\n" + "-" * 60)
    print("--- Per-Genre Error Rate (DeBERTa vs Hybrid v2 θ=0.90) ---")
    print("-" * 60)
    print(f"{'Genre':<15} {'DeBERTa Err%':>13} {'Hybrid Err%':>12} {'Delta':>8}")
    
    genres = sorted(df_enc_m['genre'].unique())
    for genre in genres:
        enc_g = df_enc_m[df_enc_m['genre'] == genre]
        hyb_g = sub[sub['genre'] == genre]
        
        enc_err = (enc_g['label_text'] != enc_g['deberta_v3_base_pred']).mean() * 100
        hyb_err = (hyb_g['label_true'] != hyb_g['label_pred']).mean() * 100 if not hyb_g.empty else 0
        
        print(f"  {genre:<13} {enc_err:>12.1f}% {hyb_err:>11.1f}% {hyb_err-enc_err:>+7.1f}%")

def main():
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    # Load data
    df_enc_m = safe_load("encoder_predictions_matched.csv")
    df_gpt4o = safe_load("api_results_gpt4o.csv")
    df_claude = safe_load("api_results_claude.csv")
    df_v2 = safe_load("hybrid_v2_results.csv")
    
    if df_enc_m.empty:
        print("Run notebooks 02-05 first to generate predictions.")
        return
        
    y_true = df_enc_m["label_text"]
    
    results = []
    
    # DeBERTa
    res = analyze_errors(y_true, df_enc_m["deberta_v3_base_pred"], "DeBERTa-v3-base")
    if res: results.append(res)
        
    # GPT-4o P4
    if not df_gpt4o.empty:
        p4 = df_gpt4o[df_gpt4o["prompt"] == "P4_few_shot_cot"]
        if len(p4) == len(y_true):
            res = analyze_errors(p4["label_true"], p4["predicted_label"], "GPT-4o P4 (CoT)")
            if res: results.append(res)
            
    # Claude — use P3 (best prompt with valid predictions)
    if not df_claude.empty:
        p3 = df_claude[(df_claude["prompt"] == "P3_few_shot") & (df_claude["predicted_label"] != "unknown")]
        if len(p3) == len(y_true):
            res = analyze_errors(p3["label_true"], p3["predicted_label"], "Claude P3 (Few-shot)")
            if res: results.append(res)
            
    # Hybrid v2
    if not df_v2.empty:
        v2 = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        if len(v2) == len(y_true):
            res = analyze_errors(v2["label_true"], v2["label_pred"], "Hybrid v2 (θ=0.90)")
            if res: results.append(res)
            
    if not results:
        return
        
    df_err = pd.DataFrame(results)
    print("\n--- Error Type Distribution (Matched Test Set) ---")
    print(df_err.to_string(index=False))
    
    df_err.to_csv(os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False)
    
    # Per-genre breakdown
    genre_error_breakdown(df_v2, df_enc_m)

    # ── Systematic error breakdown: all errors by type × genre ──
    print("\n" + "=" * 60)
    print("SYSTEMATIC ERROR BREAKDOWN — All 79 DeBERTa Errors by Genre")
    print("=" * 60)
    errors_enc = enc_m_full = df_enc_m.copy()
    err_rows = errors_enc[errors_enc["label_text"] != errors_enc["deberta_v3_base_pred"]]
    print(f"Total errors: {len(err_rows)} / {len(errors_enc)}")
    print(f"\n{'Error Type':<25} {'Count':>6} {'%':>6}")
    print("-" * 40)
    for et in [("entailment","neutral"),("entailment","contradiction"),
               ("neutral","entailment"),("neutral","contradiction"),
               ("contradiction","neutral"),("contradiction","entailment")]:
        n = ((err_rows["label_text"]==et[0]) & (err_rows["deberta_v3_base_pred"]==et[1])).sum()
        if n > 0:
            print(f"  {et[0]:>12} → {et[1]:<12} {n:>4}  ({n/len(err_rows)*100:.1f}%)")

    print(f"\n{'Genre':<15} {'Errors':>7} {'Error%':>8} {'Dominant Type':<30}")
    print("-" * 62)
    for genre in sorted(errors_enc["genre"].unique()):
        g_all  = errors_enc[errors_enc["genre"] == genre]
        g_err  = err_rows[err_rows["genre"] == genre]
        if len(g_err) == 0: continue
        # find dominant error type
        dom_type, dom_n = "", 0
        for et in [("entailment","neutral"),("neutral","entailment"),("neutral","contradiction"),
                   ("contradiction","neutral"),("entailment","contradiction"),("contradiction","entailment")]:
            n = ((g_err["label_text"]==et[0]) & (g_err["deberta_v3_base_pred"]==et[1])).sum()
            if n > dom_n:
                dom_n, dom_type = n, f"{et[0]}→{et[1]} ({n})"
        print(f"  {genre:<13} {len(g_err):>6}   {len(g_err)/len(g_all)*100:>6.1f}%   {dom_type}")

    # Sample Extraction
    print("\n" + "=" * 60)
    print("SAMPLE FAILURE CASES (Hybrid v2)")
    print("=" * 60)
    
    if not df_v2.empty:
        v2 = df_v2[(df_v2["set"] == "matched") & (df_v2["threshold"] == 0.90)]
        errors = v2[v2["label_true"] != v2["label_pred"]]
        
        # Pick one of each type
        types = [
            ("entailment", "neutral"),
            ("entailment", "contradiction"),
            ("neutral", "entailment"),
            ("contradiction", "neutral")
        ]
        
        for t_true, t_pred in types:
            subset = errors[(errors["label_true"] == t_true) & (errors["label_pred"] == t_pred)]
            if len(subset) > 0:
                sys = subset.iloc[0]
                print(f"\nType: {t_true.upper()} predicted as {t_pred.upper()}")
                print(f"Genre: {sys['genre']}")
                print(f"Preg: {sys['premise']}")
                print(f"Hyp:  {sys['hypothesis']}")
                
if __name__ == "__main__":
    main()
