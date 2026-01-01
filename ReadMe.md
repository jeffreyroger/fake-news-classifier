# Fake News Classifier


## Overview
Short project description.


## How to run locally
1. python -m venv .venv
2. source .venv/bin/activate # or .\venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. prepare data in data/processed/dataset.csv
5. python src/train_model.py
6. docker build -t fake-news-classifier .
7. docker run -p 8000:8000 fake-news-classifier


## Endpoints
POST /predict
GET /h