import pandas as pd

# Load your dataset
df = pd.read_csv("data/cleaned_news.csv")

# Check how many REAL vs FAKE examples
print(df['label'].value_counts())
