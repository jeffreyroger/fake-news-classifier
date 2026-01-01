import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models/distilbert_model"

print("MODEL DIR EXISTS:", os.path.exists(MODEL_DIR))
print("MODEL FILE EXISTS:", os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")))
print("CONFIG FILE EXISTS:", os.path.exists(os.path.join(MODEL_DIR, "config.json")))

def is_model_saved():
    return os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) and \
           os.path.exists(os.path.join(MODEL_DIR, "config.json"))

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_bert(df, text_col="text", title_col="title", label_col="label", epochs=2, batch_size=8):
    if is_model_saved():
        print(f"Found saved model in {MODEL_DIR}, loading it instead of training...")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        return model, tokenizer

    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Combine title + article
    df["combined"] = df[title_col].fillna("") + " " + df[text_col].fillna("")

    X_train, X_val, y_train, y_val = train_test_split(
        df["combined"], df[label_col],
        test_size=0.2, random_state=42, stratify=df[label_col]
    )

    train_ds = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_ds = NewsDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(DEVICE)

    # Compute class weights
    labels = df[label_col].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print("Class weights:", class_weights)

    # Loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    print("\nSaving model + tokenizer...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return model, tokenizer

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_news.csv")
    model, tokenizer = train_bert(df)
    print("Done! Model is ready for inference.")
