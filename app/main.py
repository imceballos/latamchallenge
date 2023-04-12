import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import List
from .model.model import predict_pipeline_lg

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

class PredictionOut(BaseModel):
    predicted: int

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": 1.0}

@app.post("/predict_lg", response_model=PredictionOut, tags=["predictions"])
async def predict(payload: TextIn):
    numpy_data = payload.to_string()
    predicted = await predict_pipeline_lg(numpy_data)
    return {"predicted": predicted}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=4, loop='asyncio')