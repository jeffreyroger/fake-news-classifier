from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.model_loader import init_model
from src.inference import preprocess_and_predict
from backend.schemas import PredictRequest, PredictResponse


app = FastAPI(title='Fake News Classifier')


@app.on_event('startup')
def startup_event():
    init_model()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        resp = preprocess_and_predict(req.title, req.text)
        return PredictResponse(prediction=resp['prediction'], probability=resp['probability'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)