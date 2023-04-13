import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import List
from .model.model import predict_pipeline_lg, predict_pipeline_xgb, predict_pipeline_xgb_opt

app = FastAPI()

class TextIn(BaseModel):
    data: List[float]

    @validator('data')
    def check_data_length(cls, v):
        if len(v) != 37:
            raise ValueError('Data must have 37 elements')
        return v

    def to_string(self):
        return "-".join(list((str(v) for v in self.data)))


class TextOptIn(BaseModel):
    data: List[float]

    @validator('data')
    def check_data_length(cls, v):
        if len(v) != 11:
            raise ValueError('Data must have 11 elements')
        return v

    def to_string(self):
        return "-".join(list((str(v) for v in self.data)))


class PredictionOut(BaseModel):
    predicted: int

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": 1.0}

@app.post("/predict_lg", response_model=PredictionOut, tags=["predictions"])
async def predict_lg(payload: TextIn):
    numpy_data = payload.to_string()
    predicted = await predict_pipeline_lg(numpy_data)
    return {"predicted": predicted}

@app.post("/predict_xgb", response_model=PredictionOut, tags=["predictions"])
async def predict_xgb(payload: TextIn):
    numpy_data = payload.to_string()
    predicted = await predict_pipeline_xgb(numpy_data)
    return {"predicted": predicted}

@app.post("/predict_xgb_opt", response_model=PredictionOut, tags=["predictions"])
async def predict_xgb_opt(payload: TextOptIn):
    numpy_data = payload.to_string()
    predicted = await predict_pipeline_xgb_opt(numpy_data)
    return {"predicted": predicted}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=4, loop='asyncio')