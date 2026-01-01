import joblib
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import clean_text, tokenize_and_lemmatize
from sklearn.preprocessing import StandardScaler
from src.preprocessing import clean_text, tokenize_and_lemmatize, extract_basic_metadata
from src.feature_engineering import fit_vectorizer, transform_features


def train_pipeline(df: pd.DataFrame, text_col='text', title_col='title', label_col='label'):

    print(f"🔄 Cleaning and preprocessing using columns: text='{text_col}', title='{title_col}', label='{label_col}'...")

    # Fill title if missing
    if title_col not in df.columns:
        print(f"⚠️ Title column '{title_col}' not found, using empty string.")
        df[title_col] = ""

    # 1. CLEAN + TOKENIZE (consistent with prediction)
    df['text_clean'] = df[text_col].fillna('').apply(lambda x: tokenize_and_lemmatize(clean_text(str(x))))
    df['title_clean'] = df[title_col].fillna('').apply(lambda x: tokenize_and_lemmatize(clean_text(str(x))))

    # 2. COMBINE TITLE + TEXT
    combined_texts = (df['title_clean'] + ' ' + df['text_clean']).tolist()

    # 3. METADATA
    print("📊 Extracting metadata...")
    meta_list = []
    for idx, row in df.iterrows():
        meta = extract_basic_metadata(row[title_col], row[text_col])
        meta_list.append(list(meta.values()))
    meta_df = pd.DataFrame(meta_list)

    # 4. FIT VECTORIZER AND SCALER
    print("🔤 Fitting vectorizer...")
    vec = fit_vectorizer(combined_texts)
    
    print("📏 Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(meta_df)

    # 5. TRANSFORM FEATURES
    print("🔢 Transforming features...")
    X = transform_features(vec, scaler, combined_texts, meta_df)
    y = df[label_col].astype(int).values

    # 6. TRAIN/VAL SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. TRAIN MODEL
    print("🤖 Training Logistic Regression...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # 8. SAVE ARTIFACTS
    print("💾 Saving model + vectorizer + scaler...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(vec, 'models/vectorizer.pkl')
    joblib.dump(scaler, 'models/metadata_scaler.pkl')

    # 9. EVALUATION
    print("📊 Evaluation:")
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))
    print(confusion_matrix(y_val, preds))

    print("✅ Training complete!")
    return model

