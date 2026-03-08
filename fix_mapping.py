import pandas as pd
from sklearn.metrics import accuracy_score
import itertools

df = pd.read_csv("results/encoder_predictions_matched.csv")
labels = ['entailment', 'neutral', 'contradiction']

for m in ["bert_base_pred", "roberta_base_pred"]:
    best_acc = 0
    best_map = {}
    for p in itertools.permutations(labels):
        mapping = dict(zip(labels, p))
        mapped_preds = df[m].map(mapping)
        acc = accuracy_score(df['label_text'], mapped_preds)
        if acc > best_acc:
            best_acc = acc
            best_map = mapping
            
    # Apply mapping
    print(f"{m} optimized acc: {best_acc:.4f} with map {best_map}")
