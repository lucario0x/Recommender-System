import os, glob, zipfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

from config import ZIP_PATH, ROOT, N_ITEMS, MIN_PER_TYPE, MIN_IN_SAMPLE, SEED


@dataclass
class Catalogue:
    df: pd.DataFrame
    paths: list
    labels: np.ndarray          # article type, the main relevance label
    label_sets: dict            # alternative relevance definitions
    type_names: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def find_data_dir():
    if not glob.glob(f"{ROOT}/**/styles.csv", recursive=True):
        assert os.path.exists(ZIP_PATH), f"zip not found at {ZIP_PATH}"
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(ROOT)
    return os.path.dirname(glob.glob(f"{ROOT}/**/styles.csv", recursive=True)[0])


def load_catalogue():
    data_dir = find_data_dir()
    df = pd.read_csv(f"{data_dir}/styles.csv", on_bad_lines="skip")
    df = df.dropna(subset=["articleType", "baseColour", "masterCategory"])
    df = df[[os.path.exists(f"{data_dir}/images/{i}.jpg") for i in df["id"]]]
    counts = df["articleType"].value_counts()
    df = df[df["articleType"].isin(counts[counts >= MIN_PER_TYPE].index)]
    df = df.sample(min(N_ITEMS, len(df)), random_state=SEED)
    counts = df["articleType"].value_counts()
    df = df[df["articleType"].isin(counts[counts >= MIN_IN_SAMPLE].index)].reset_index(drop=True)

    paths = [f"{data_dir}/images/{i}.jpg" for i in df["id"]]
    codes, names = pd.factorize(df["articleType"])
    label_sets = {
        "master_category": pd.factorize(df["masterCategory"])[0],
        "article_type": codes,
        "type+colour": pd.factorize(df["articleType"] + "|" + df["baseColour"])[0],
    }
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=codes)
    print(f"{len(df)} products, {len(names)} types, train {len(train_idx)} / query {len(test_idx)}")
    return Catalogue(df, paths, codes, label_sets, np.asarray(names), train_idx, test_idx)


def load_small(paths, size=64):
    # low-res tensors for the from-scratch models
    ds = ImgDS(paths, T.Compose([T.Resize((size, size)), T.ToTensor()]))
    return torch.cat(list(torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2)))
