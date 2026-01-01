import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models/distilbert_model"

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
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DistilBERTFakeNews:
    def __init__(self, model_dir=MODEL_DIR, load_model=True, df=None, epochs=2):
        self.model_dir = model_dir

        if load_model and os.path.exists(model_dir):
            print(f"Loading saved model from {model_dir}...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.model.to(DEVICE)
            self.model.eval()
        else:
            if df is None:
                raise ValueError("Training data required if model is not already saved.")
            self.train(df, epochs)

    def train(self, df, epochs=2, text_col="text", title_col="title", label_col="label"):
        print("Training DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        df["combined"] = df[title_col].fillna("") + " " + df[text_col].fillna("")

        X_train, X_val, y_train, y_val = train_test_split(
            df["combined"], df[label_col],
            test_size=0.2, random_state=42, stratify=df[label_col]
        )

        train_ds = NewsDataset(X_train.tolist(), y_train.tolist(), self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.model.to(DEVICE)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        self.model.eval()
        print(f"Model saved to {self.model_dir}")

    def predict(self, title, text):
        combined_text = (title or "") + " " + (text or "")
        encoded = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        return "REAL" if pred == 1 else "FAKE"
