import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "models/distilbert_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

tests = [
    ("This is a completely fake made-up story that never happened.", "FAKE expected"),
    ("The government of India announced a new policy today during a press conference.", "REAL expected"),
    ("Aliens landed on Earth and took over the White House.", "FAKE expected"),
    ("Google released a new AI model today after months of research.", "REAL expected"),
]

for text, expected in tests:
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    input_ids = encoded["input_ids"].to(DEVICE)
    mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=mask)
        pred = torch.argmax(logits.logits, dim=1).item()

    label = "REAL" if pred == 1 else "FAKE"

    print(f"\nInput: {text}")
    print(f"Prediction: {label} | {expected}")
