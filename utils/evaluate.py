import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("results/encoder_predictions_matched.csv")
models = ["bert_base_pred", "deberta_v3_small_pred", "roberta_base_pred", "deberta_v3_base_pred"]
for m in models:
    acc = accuracy_score(df['label_text'], df[m])
    print(f"{m} -> ACC: {acc:.4f}")
    
df_mm = pd.read_csv("results/encoder_predictions_mm.csv")
for m in models:
    acc = accuracy_score(df_mm['label_text'], df_mm[m])
    print(f"MM {m} -> ACC: {acc:.4f}")
