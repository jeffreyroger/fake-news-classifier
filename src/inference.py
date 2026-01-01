import joblib
import numpy as np
import os
import torch
from src.preprocessing import clean_text, tokenize_and_lemmatize, extract_basic_metadata
from scipy.sparse import hstack
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL = None
VECTORIZER = None
SCALER = None
BERT_MODEL = None
BERT_TOKENIZER = None
MODEL_TYPE = os.environ.get("MODEL_TYPE", "baseline")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_artifacts(model_path='models/model.pkl', vec_path='models/vectorizer.pkl', scaler_path='models/metadata_scaler.pkl', bert_dir='models/distilbert_model'):
    global MODEL, VECTORIZER, SCALER, BERT_MODEL, BERT_TOKENIZER, MODEL_TYPE
    
    if MODEL_TYPE == "bert":
        print(f"Loading BERT model from {bert_dir}...")
        BERT_TOKENIZER = DistilBertTokenizer.from_pretrained(bert_dir)
        BERT_MODEL = DistilBertForSequenceClassification.from_pretrained(bert_dir)
        BERT_MODEL.to(DEVICE)
        BERT_MODEL.eval()
    else:
        print("Loading Baseline model artifacts...")
        MODEL = joblib.load(model_path)
        VECTORIZER = joblib.load(vec_path)
        SCALER = joblib.load(scaler_path)

def predict_baseline(title, text):
    # Clean and tokenize
    text_clean = tokenize_and_lemmatize(clean_text(text))
    title_clean = tokenize_and_lemmatize(clean_text(title))

    # Extract metadata
    metadata = extract_basic_metadata(title, text)

    # Vectorize text
    combined = title_clean + ' ' + text_clean
    X_text = VECTORIZER.transform([combined])

    # Scale metadata
    meta_arr = np.array([list(metadata.values())])
    meta_scaled = SCALER.transform(meta_arr)

    # Combine text and metadata features
    X = hstack([X_text, meta_scaled])

    # Predict
    proba = MODEL.predict_proba(X)[0]
    pred = MODEL.predict(X)[0]

    return {'prediction': int(pred), 'probability': proba.tolist()}

def predict_bert(title, text):
    combined_text = (title or "") + " " + (text or "")
    encoded = BERT_TOKENIZER(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = BERT_MODEL(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = torch.argmax(logits, dim=1).item()

    return {'prediction': int(pred), 'probability': probs.tolist()}

def preprocess_and_predict(title: str, text: str):
    if MODEL_TYPE == "bert":
        return predict_bert(title, text)
    else:
        return predict_baseline(title, text)

