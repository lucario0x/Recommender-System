import os
import numpy as np
import torch

ZIP_PATH = "/content/archive.zip"
ROOT = "/content/fashion"

N_ITEMS = 5000
MIN_PER_TYPE = 30      # drop very rare article types (full dataset)
MIN_IN_SAMPLE = 5      # and again after sampling, so every type can be split/retrieved
K = 10
SEED = 0
AE_EPOCHS = 40
VIT_EPOCHS = 60

CACHE = "cache"
OUT = "outputs"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
