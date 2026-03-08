import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

data_dir = "results"

print("--- SECTION 2 ---")
enc_m = pd.read_csv(f"{data_dir}/encoder_predictions_matched.csv")
enc_mm = pd.read_csv(f"{data_dir}/encoder_predictions_mm.csv")

mappings = {
    "bert_base": {'entailment': 'contradiction', 'neutral': 'entailment', 'contradiction': 'neutral'},
    "roberta_base": {'entailment': 'contradiction', 'neutral': 'neutral', 'contradiction': 'entailment'},
    "deberta_v3_small": None,
    "deberta_v3_base": None
}

for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
    pred = enc_m[f"{model}_pred"]
    if mappings[model]:
        pred = pred.map(mappings[model])
        
    acc = accuracy_score(enc_m['label_text'], pred)
    macro = f1_score(enc_m['label_text'], pred, average='macro')
    # calculate ent-f1, neu-f1, con-f1
    labels = list(enc_m['label_text'].unique())
    f1s = f1_score(enc_m['label_text'], pred, average=None, labels=['entailment', 'neutral', 'contradiction'])
    ent_f1, neu_f1, con_f1 = f1s
    print(f"| {model} | {acc*100:.1f}% | {macro:.4f} | {ent_f1:.4f} | {neu_f1:.4f} | {con_f1:.4f} |")

print("--- SECTION 2 (Mismatched) ---")
for model in ["bert_base", "deberta_v3_small", "roberta_base", "deberta_v3_base"]:
    pred_m = enc_m[f"{model}_pred"]
    pred_mm = enc_mm[f"{model}_pred"]
    
    if mappings[model]:
        pred_m = pred_m.map(mappings[model])
        pred_mm = pred_mm.map(mappings[model])
        
    acc_m = accuracy_score(enc_m['label_text'], pred_m)
    acc_mm = accuracy_score(enc_mm['label_text'], pred_mm)
    print(f"| {model} | {acc_mm*100:.1f}% | {(acc_m - acc_mm)*100:.1f}% |")

print("--- SECTION 3 (GPT-4o) ---")
g = pd.read_csv(f"{data_dir}/api_results_gpt4o.csv")
g_mm = pd.concat([pd.read_csv(f"{data_dir}/api_results_gpt4o_mm.csv")])

for p in ["P1_zero_shot", "P2_zero_shot_def", "P3_few_shot", "P4_few_shot_cot"]:
    sub_m = g[g['prompt'] == p]
    sub_mm = g_mm[g_mm['prompt'] == p]
    acc_m = accuracy_score(sub_m['label_true'], sub_m['predicted_label'])
    acc_mm = accuracy_score(sub_mm['label_true'], sub_mm['predicted_label'])
    avg_tokens = (sub_m['prompt_tokens'].mean() + sub_m['completion_tokens'].mean())
    c = sub_m['cost_usd'].sum() / (len(sub_m) / 1000)
    print(f"| {p} | {acc_m*100:.1f}% | {acc_mm*100:.1f}% | {(acc_m - acc_mm)*100:.1f}% | {avg_tokens:.0f} | ${c:.3f} |")
    
print("--- SECTION 4 (Other LLMs) ---")
for model, csv in [("GPT-5", "api_results_gpt5.csv"), ("Claude Sonnet 4.5", "api_results_claude.csv")]:
    other = pd.read_csv(f"{data_dir}/{csv}")
    print(f"\n{model}:")
    for p in ["P1_zero_shot", "P2_zero_shot_def", "P3_few_shot", "P4_few_shot_cot"]:
        sub_m = other[other['prompt'] == p]
        if len(sub_m) > 0:
            acc_m = accuracy_score(sub_m['label_true'], sub_m['predicted_label'])
            print(f"| {p} | {acc_m*100:.2f}% | {len(sub_m)}/800 |")

print("\n--- SECTION 5 (Hybrid Architectures) ---")
for v, csv in [("v1", "hybrid_v1_results.csv"), ("v2", "hybrid_v2_results.csv"), ("v3", "hybrid_v3_results.csv"), ("v4", "hybrid_v4_results.csv"), ("v5", "hybrid_v5_results.csv")]:
    try:
        df_v = pd.read_csv(f"{data_dir}/{csv}")
        print(f"\nHybrid {v}:")
        if "threshold" in df_v.columns:
            for th in [0.85, 0.90, 0.95]:
                sub = df_v[(df_v['set'] == 'matched') & (df_v['threshold'] == th)]
                if len(sub) > 0:
                    acc = accuracy_score(sub['label_true'], sub['label_pred'])
                    api_pct = (sub['source'] == 'api').mean() * 100
                    print(f"| θ={th} | {acc*100:.2f}% | API: {api_pct:.1f}% |")
        else:
            # v5 ensemble gate doesn't have a threshold column
            sub = df_v[df_v['set'] == 'matched']
            acc = accuracy_score(sub['label_true'], sub['label_pred'])
            api_pct = (sub['source'] == 'api').mean() * 100
            print(f"| Ensemble | {acc*100:.2f}% | API: {api_pct:.1f}% |")
    except:
        continue
