import os
from src.inference import load_artifacts




def init_model():
    # Paths are relative to the project root
    load_artifacts(
        model_path='models/model.pkl',
        vec_path='models/vectorizer.pkl',
        scaler_path='models/metadata_scaler.pkl',
        bert_dir='models/distilbert_model'
    )