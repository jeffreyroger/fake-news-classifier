import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)    # keep alphabets
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

if __name__ == "__main__":
    print("Loading datasets...")

    true_df = pd.read_csv("raw/True.csv")
    fake_df = pd.read_csv("raw/Fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)

    print("Preprocessing text...")
    df["text"] = df["text"].apply(preprocess)

    df.to_csv("cleaned_news.csv", index=False)

    print("Done! Saved cleaned_news.csv")
