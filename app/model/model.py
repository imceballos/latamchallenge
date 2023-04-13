from concurrent.futures import ThreadPoolExecutor
from pathlib import Path 
import asyncio
import pickle
from functools import lru_cache
import xgboost
import aiofiles

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

async def load_lg_model():
    async with aiofiles.open(f"{BASE_DIR}/lr_model.pkl", mode='rb') as f:
        model_lg = pickle.loads(await f.read())
    return model_lg

async def load_xgb_model():
    async with aiofiles.open(f"{BASE_DIR}/xgb_model.pkl", mode='rb') as f:
        model_xgb = pickle.loads(await f.read())
    return model_xgb

async def load_xgbopt_model():
    async with aiofiles.open(f"{BASE_DIR}/xgb_model_opt.pkl", mode='rb') as f:
        model_xgb_opt = pickle.loads(await f.read())
    return model_xgb_opt

model_lg = asyncio.run(load_lg_model())
model_xgb = asyncio.run(load_xgb_model())
model_xgb_opt = asyncio.run(load_xgbopt_model())

@lru_cache(maxsize=512)
def predict_lg(text):
    integers = (int(float(v)) for v in text.split("-"))
    return model_lg.predict([list(integers)])[0]

@lru_cache(maxsize=512)
def predict_xgb(text):
    integers = (int(float(v)) for v in text.split("-"))
    return model_xgb.predict([list(integers)])[0]

@lru_cache(maxsize=512)
def predict_xgb_opt(text):
    integers = (int(float(v)) for v in text.split("-"))
    return model_xgb_opt.predict([list(integers)])[0]

async def predict_pipeline_lg(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        pred_coro = loop.run_in_executor(pool, predict_lg, text)
        pred = await pred_coro
    return pred

async def predict_pipeline_xgb(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        pred_coro = loop.run_in_executor(pool, predict_xgb, text)
        pred = await pred_coro
    return pred

async def predict_pipeline_xgb_opt(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        pred_coro = loop.run_in_executor(pool, predict_xgb_opt, text)
        pred = await pred_coro
    return pred