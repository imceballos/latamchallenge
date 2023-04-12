from concurrent.futures import ThreadPoolExecutor
from pathlib import Path 
import asyncio
import pickle
from functools import lru_cache
import aiofiles

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

async def load_model():
    async with aiofiles.open(f"{BASE_DIR}/lr_model.pkl", mode='rb') as f:
        model_lg = pickle.loads(await f.read())
    return model_lg

model_lg = asyncio.run(load_model())

@lru_cache(maxsize=512)
def predict(text):
    integers = (int(float(v)) for v in text.split("-"))
    return model_lg.predict([list(integers)])[0]

async def predict_pipeline_lg(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        pred_coro = loop.run_in_executor(pool, predict, text)
        pred = await pred_coro
    return pred
