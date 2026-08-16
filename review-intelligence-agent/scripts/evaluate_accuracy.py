import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from utils.model_loader import get_sentiment_pipeline

# 1. Annotated Ground Truth for the sample reviews dataset
ground_truth_labels = [
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE",
    "POSITIVE"
]

csv_path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "sample_reviews.csv")
df = pd.read_csv(csv_path)
df["ground_truth"] = ground_truth_labels[:len(df)]

models = ["DistilBERT", "RoBERTa", "DeBERTa"]
results = {}

print("=" * 60)
print(f"EVALUATING MODEL ACCURACY ON {len(df)} DATASET SAMPLES")
print("=" * 60)

for model_name in models:
    print(f"\nEvaluating {model_name}...")
    pipe = get_sentiment_pipeline(model_name)
    
    preds = []
    confidences = []
    
    raw_outputs = pipe(df["review"].tolist())
    
    for out in raw_outputs:
        lbl = out["label"].upper()
        # Map labels to binary POSITIVE/NEGATIVE
        if "POS" in lbl:
            pred_lbl = "POSITIVE"
        elif "NEG" in lbl:
            pred_lbl = "NEGATIVE"
        else:
            pred_lbl = "NEUTRAL"
        preds.append(pred_lbl)
        confidences.append(out["score"])
        
    df[f"pred_{model_name}"] = preds
    
    # Filter out neutral for binary accuracy comparison or treat as mismatch
    # Direct Accuracy:
    matches = sum(1 for p, g in zip(preds, df["ground_truth"]) if p == g)
    acc = matches / len(df) * 100.0
    avg_conf = np.mean(confidences) * 100.0
    
    results[model_name] = {
        "Accuracy (%)": round(acc, 2),
        "Correct": matches,
        "Total": len(df),
        "Avg Confidence (%)": round(avg_conf, 2)
    }

print("\n" + "=" * 60)
print("FINAL ACCURACY RESULTS SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame(results).T
print(summary_df.to_string())
