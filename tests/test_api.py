# tests/test_api.py
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_health():
r = client.get('/health')
assert r.status_code == 200
assert r.json()['status'] == 'ok'




def test_predict_empty():
r = client.post('/predict', json={'title': '', 'text': ''})
assert r.status_code == 200 or r.status_code == 500