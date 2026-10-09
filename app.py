# uvicorn app:app --port 8000   ->   http://localhost:8000/docs
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

E = np.load("outputs/emb_best.npy")
CAT = pd.read_csv("outputs/catalogue.csv")
POS = {pid: i for i, pid in enumerate(CAT["id"])}
app = FastAPI(title="fashion recommender")


@app.get("/similar/{product_id}")
def similar(product_id: int, k: int = 10):
    if product_id not in POS:
        raise HTTPException(404, "unknown product")
    t0 = time.perf_counter()
    i = POS[product_id]
    s = E @ E[i]
    s[i] = -1                                  # don't return the query itself
    top = np.argsort(-s)[:k]
    return {"query": CAT.iloc[i].to_dict(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "results": [{**CAT.iloc[j].to_dict(), "score": float(s[j])} for j in top]}
