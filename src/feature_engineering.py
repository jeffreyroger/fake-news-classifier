import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def fit_vectorizer(texts, max_features=5000):
    """Fit TF-IDF vectorizer on combined text."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2
    )
    vec.fit(texts)
    return vec


def transform_features(vec, scaler, texts, metadata_df: pd.DataFrame):
    """Transform text → TF-IDF + scale metadata using already fitted objects."""
    
    # TF-IDF vectors
    X_text = vec.transform(texts)
    
    # Scale metadata
    X_meta = scaler.transform(metadata_df)

    # Combine sparse (TFIDF) + dense metadata
    from scipy.sparse import hstack
    X = hstack([X_text, X_meta])

    return X

