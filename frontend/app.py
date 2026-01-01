# frontend/app.py

import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import load_artifacts, preprocess_and_predict

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="wide")

st.title("📰 Fake News Classifier")

# Model Selection
model_type = st.sidebar.selectbox("Select Model", ["baseline", "bert"], index=0)
os.environ["MODEL_TYPE"] = model_type

# Load artifacts (cached to avoid reloading on every interaction)
@st.cache_resource
def init_inference(m_type):
    load_artifacts()

init_inference(model_type)

st.write(f"Currently using: **{model_type.upper()}** model.")
st.write("Enter a news title and article to check if it is REAL or FAKE.")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("News Title")
    text = st.text_area("News Text", height=300)

if st.button("Predict", type="primary"):
    if not text:
        st.warning("Please enter at least the news text!")
    else:
        with st.spinner("Analyzing..."):
            res = preprocess_and_predict(title, text)
            
            # Based on training on data/processed/train.csv: 0 is REAL, 1 is FAKE
            prediction = "REAL" if res['prediction'] == 0 else "FAKE"
            prob_real = res['probability'][0]
            prob_fake = res['probability'][1]
            
            st.subheader(f"Prediction: {prediction}")
            
            # Progress bars for probabilities
            st.write(f"REAL Probability: {prob_real:.2%}")
            st.progress(prob_real)
            
            st.write(f"FAKE Probability: {prob_fake:.2%}")
            st.progress(prob_fake)

            
            if prediction == "REAL":
                st.success("This news article appears to be REAL.")
            else:
                st.error("This news article appears to be FAKE.")

