from concurrent.futures import ThreadPoolExecutor
from pathlib import Path 
import redis
import asyncio
import pickle
import re
import time
from functools import lru_cache

#r = redis.Redis(
#  host='redis-14663.c1.us-central1-2.gce.cloud.redislabs.com',
#  port=14663,
#  password='Hi1ZjwRp0WmrUhGK3HqpdCnIunrHHiEy')

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/lr_model.pkl", "rb") as f:
    model_lg = pickle.load(f)


@lru_cache(maxsize=2048)
def predict(text):
    integers = (int(float(v)) for v in text.split("-"))
    return model_lg.predict([list(integers)])[0]
    #return model.predict([list(map(int, map(float, text.split("-"))))])[0]
    #return model.predict([[int(float(v)) for v in text.split("-")]])[0]

async def predict_pipeline_lg(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        pred_coro = loop.run_in_executor(pool, predict, text)
        pred = await pred_coro
        #r.set(n_text, str(pred[0]))
    #pred = await predict(text)
    #print("--- %s seconds ---" % (time.time() - start_time))
    return pred
