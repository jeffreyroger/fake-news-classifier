import torch
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BERTFakeNews:
    def __init__(self, model_path="models/bert_model"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, title, article):
        text = (title or "") + " " + article

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            output = self.model(input_ids, mask)
            logits = output.logits
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

        label = prob.argmax()
        fake_prob = float(prob[1])

        return label, fake_prob
