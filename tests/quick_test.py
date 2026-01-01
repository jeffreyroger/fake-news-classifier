import os
import sys
import pandas as pd

sys.path.append(os.getcwd())

from src.inference import load_artifacts, preprocess_and_predict

# Load baseline model
os.environ["MODEL_TYPE"] = "baseline"
load_artifacts()

# Sample 10 random articles from test set
df = pd.read_csv('data/processed/test.csv').sample(10, random_state=42)

correct = 0
print("Testing 10 random samples from test set:\n")

for idx, row in df.iterrows():
    res = preprocess_and_predict('', row['content'])
    pred = res['prediction']
    actual = row['label']
    match = '✓' if pred == actual else '✗'
    
    label_name_actual = "REAL" if actual == 0 else "FAKE"
    label_name_pred = "REAL" if pred == 0 else "FAKE"
    
    print(f"{match} Actual: {label_name_actual} | Predicted: {label_name_pred} | Confidence: {max(res['probability']):.2%}")
    correct += (pred == actual)

print(f"\n{'='*50}")
print(f"Accuracy: {correct}/10 = {correct/10*100:.1f}%")
