import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "models/distilbert_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
print("🔍 Loading model from:", MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully on:", DEVICE, "\n")

CONFIDENCE_THRESHOLD = 0.8  # Adjust between 0.7-0.9 as you see fit

def predict_news(title, text):
    combined_text = (title or "") + " " + (text or "")

    encoded = tokenizer(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = "REAL" if probs[1] > probs[0] else "FAKE"
    confidence = max(probs)

    if confidence < CONFIDENCE_THRESHOLD:
        return "UNSURE", probs[0], probs[1]

    return pred_label, probs[0], probs[1]


if __name__ == "__main__":
    title = input("Enter news title: ")
    text = input("Enter news text: ")

    result, fake_prob, real_prob = predict_news(title, text)
    print(f"\nPrediction: {result} (Confidence: {max(fake_prob, real_prob):.2f})")
    print(f"Probabilities → FAKE: {fake_prob:.2f}, REAL: {real_prob:.2f}")
