import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.inference import load_artifacts, preprocess_and_predict

def evaluate_on_test_data(test_csv_path='data/processed/test.csv'):
    print(f"Loading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    
    # The columns in test.csv are ['label', 'content', 'clean_text']
    # Our inference expects title and text. 
    # Since we only have 'content', we'll pass it as text and empty string as title.
    
    print("Loading Baseline model...")
    os.environ["MODEL_TYPE"] = "baseline"
    load_artifacts()
    
    y_true = df['label'].astype(int).tolist()
    y_pred = []
    
    print(f"Running inference on {len(df)} rows...")
    # Using a small sample if it's too slow, but let's try the first 500 for a quick check
    sample_size = min(500, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    for idx, row in df_sample.iterrows():
        # Based on previous analysis, we used title + text
        # If the test set only has 'content', we'll map that.
        res = preprocess_and_predict(title="", text=row['content'])
        y_pred.append(res['prediction'])
        
    y_true_sample = df_sample['label'].astype(int).tolist()
    
    print("\n--- Baseline Model Performance on User Test Data ---")
    print(classification_report(y_true_sample, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_sample, y_pred))

if __name__ == "__main__":
    evaluate_on_test_data()
