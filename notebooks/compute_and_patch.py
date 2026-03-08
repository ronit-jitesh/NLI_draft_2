#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import re

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
REPORT_PATH = os.path.join(PROJECT_DIR, "NLI_Comprehensive_Results.md")

L = ['entailment', 'neutral', 'contradiction']

def get_row_metrics(y_true, y_pred, name):
    y_true_clean = []
    y_pred_clean = []
    for t, p in zip(y_true, y_pred):
        if p in L:
            y_true_clean.append(t)
            y_pred_clean.append(p)
    
    p, r, f1, _ = precision_recall_fscore_support(y_true_clean, y_pred_clean, labels=L, zero_division=0)
    return [name] + [f"{p[i]:.3f} / {r[i]:.3f} / {f1[i]:.3f}" for i in range(3)]

def patch_report():
    with open(REPORT_PATH, 'r') as f:
        report = f.read()

    # --- 1. SECTION 2.1 Encoders ---
    df_enc_m = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv"))
    y_true_m = df_enc_m["label_text"]
    
    rows = []
    rows.append(get_row_metrics(y_true_m, df_enc_m["bert_base_pred"], "BERT-base"))
    rows.append(get_row_metrics(y_true_m, df_enc_m["deberta_v3_small_pred"], "DeBERTa-v3-small"))
    rows.append(get_row_metrics(y_true_m, df_enc_m["roberta_base_pred"], "RoBERTa-base"))
    rows.append(get_row_metrics(y_true_m, df_enc_m["deberta_v3_base_pred"], "DeBERTa-v3-base"))
    rows.append(get_row_metrics(y_true_m, df_enc_m["deberta_v3_large_pred"], "DeBERTa-v3-large"))

    enc_table = "| Model | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |\n"
    enc_table += "|-------|----------------|----------------|----------------|\n"
    for r in rows:
        enc_table += f"| {r[0]:<16} | {r[1]} | {r[2]} | {r[3]} |\n"

    # Robust Replace Section 2.1 Table
    # Match any table starting with | Model | and having BERT-base to DeBERTa-v3-large
    report = re.sub(r"\| Model \|.*?DeBERTa-v3-large \|.*?\n", enc_table, report, count=1, flags=re.DOTALL)

    # --- 2. SECTION 3.3.1 GPT-4o ---
    df_gpt4o = pd.read_csv(os.path.join(RESULTS_DIR, "api_results_gpt4o.csv"))
    rows = []
    for p in ["P1_zero_shot", "P2_zero_shot_def", "P3_few_shot", "P4_few_shot_cot"]:
        sub = df_gpt4o[df_gpt4o["prompt"] == p]
        if not sub.empty:
            rows.append(get_row_metrics(sub["label_true"], sub["predicted_label"], p))
    
    gpt_table = "| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |\n"
    gpt_table += "|--------|----------------|----------------|----------------|\n"
    for r in rows:
        gpt_table += f"| {r[0]:<16} | {r[1]} | {r[2]} | {r[3]} |\n"

    # Match existing GPT table block
    report = re.sub(r"### 3\.3\.1 Per-Class Metrics.*?### 3\.4", "### 3.3.1 Per-Class Metrics (P / R / F1)\n\n" + gpt_table + "\n### 3.4", report, flags=re.DOTALL)

    # --- 3. SECTION 4.2.1 Claude ---
    df_claude = pd.read_csv(os.path.join(RESULTS_DIR, "api_results_claude.csv"))
    rows = []
    for p in ["P1_zero_shot", "P2_zero_shot_def", "P3_few_shot"]:
        sub = df_claude[df_claude["prompt"] == p]
        if not sub.empty:
            rows.append(get_row_metrics(sub["label_true"], sub["predicted_label"], p))
    
    claude_table = "| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |\n"
    claude_table += "|--------|----------------|----------------|----------------|\n"
    for r in rows:
        claude_table += f"| {r[0]:<16} | {r[1]} | {r[2]} | {r[3]} |\n"

    report = re.sub(r"### 4\.2\.1 Per-Class Metrics.*?### 4\.3", "### 4.2.1 Per-Class Metrics (P / R / F1)\n\n" + claude_table + "\n### 4.3", report, flags=re.DOTALL)

    # --- 4. SECTION 5.5.1 Hybrid ---
    hyb_v1 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v1_results.csv"))
    hyb_v2 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v2_results.csv"))
    hyb_v4 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v4_results.csv"))
    hyb_v5 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v5_results.csv"))
    
    rows = []
    rows.append(get_row_metrics(y_true_m, df_enc_m["deberta_v3_base_pred"], "DeBERTa-base"))
    v1 = hyb_v1[(hyb_v1["set"]=="matched") & (hyb_v1["threshold"]==0.9)]
    if not v1.empty: rows.append(get_row_metrics(v1["label_true"], v1["label_pred"], "Hybrid v1 (0.9)"))
    v2 = hyb_v2[(hyb_v2["set"]=="matched") & (hyb_v2["threshold"]==0.9)]
    if not v2.empty: rows.append(get_row_metrics(v2["label_true"], v2["label_pred"], "Hybrid v2 (0.9)"))
    v4 = hyb_v4[(hyb_v4["set"]=="matched") & (hyb_v4["threshold"]==0.9)]
    if not v4.empty: rows.append(get_row_metrics(v4["label_true"], v4["label_pred"], "Hybrid v4 (0.9)"))
    if not hyb_v5.empty: rows.append(get_row_metrics(hyb_v5["label_true"], hyb_v5["label_pred"], "Hybrid v5 Ens"))

    hyb_table = "| System | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |\n"
    hyb_table += "|--------|----------------|----------------|----------------|\n"
    for r in rows:
        hyb_table += f"| {r[0]:<16} | {r[1]} | {r[2]} | {r[3]} |\n"

    report = re.sub(r"### 5\.5\.1 Per-Class Metrics.*?### 5\.6", "### 5.5.1 Per-Class Metrics (P / R / F1)\n\n" + hyb_table + "\n### 5.6", report, flags=re.DOTALL)

    # --- 5. SECTION 8.1 Systematic Genre Breakdown ---
    err_rows = df_enc_m[df_enc_m["label_text"] != df_enc_m["deberta_v3_base_pred"]]
    breakdown = "| Genre | Errors | Error % | Dominant Error Type |\n"
    breakdown += "|-------|--------|---------|---------------------|\n"
    for genre in sorted(df_enc_m["genre"].unique()):
        g_all = df_enc_m[df_enc_m["genre"] == genre]
        g_err = err_rows[err_rows["genre"] == genre]
        dom_type, dom_n = "N/A", 0
        from itertools import permutations
        for t1, t2 in permutations(L, 2):
            n = ((g_err["label_text"]==t1) & (g_err["deberta_v3_base_pred"]==t2)).sum()
            if n > dom_n:
                dom_n, dom_type = n, f"{t1}->{t2}"
        
        breakdown += f"| {genre:<12} | {len(g_err):>6} | {len(g_err)/len(g_all)*100:>7.1f}% | {dom_type:<19} |\n"

    report = re.sub(r"### 8\.1 Systematic Error Breakdown by Genre.*?### 8\.2", "### 8.1 Systematic Error Breakdown by Genre\n\n" + breakdown + "\n### 8.2", report, flags=re.DOTALL)

    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print("Report surgically patched with rubric-aligned metrics (ROBUST).")

if __name__ == "__main__":
    patch_report()
