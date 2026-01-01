from pydantic import BaseModel


class PredictRequest(BaseModel):
    title: str | None = None
    text: str


class PredictResponse(BaseModel):
    prediction: int
    probability: list[float]