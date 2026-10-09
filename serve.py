import numpy as np
from config import OUT
from metrics import l2norm


def save_index(emb, cat):
    # files read by app.py
    np.save(f"{OUT}/emb_best.npy", l2norm(emb.astype(np.float32)))
    cat.df[["id", "articleType", "baseColour", "productDisplayName"]].to_csv(f"{OUT}/catalogue.csv", index=False)
