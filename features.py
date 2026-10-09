import os, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm
from torchvision import transforms as T

from config import CACHE, device
from data import ImgDS

imagenet = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
prep224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), imagenet])


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def extract(name, forward, tf, paths, bs=64):
    f = f"{CACHE}/{name}.npy"
    if os.path.exists(f):                       # reuse earlier run (time is then unknown)
        return np.load(f), None
    dl = torch.utils.data.DataLoader(ImgDS(paths, tf), batch_size=bs, num_workers=2)
    out = []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device, enabled=(device == "cuda")):
        for x in tqdm(dl, desc=name):
            out.append(forward(x.to(device)).float().cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    emb = torch.cat(out).numpy()
    np.save(f, emb)
    return emb, time.time() - t0


def gem_pool(f, p=3.0):
    # generalised-mean pooling: p=1 is average pooling, large p approaches max pooling
    return f.clamp(min=1e-6).pow(p).mean((2, 3)).pow(1 / p)


def resnet_embeddings(paths, emb, cost):
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    body = nn.Sequential(*list(m.children())[:-2])          # everything except avgpool + fc
    for name, key, fn in [("ResNet-50 (avg pool)", "resnet50_avg", lambda x: body(x).mean((2, 3))),
                          ("ResNet-50 (GeM pool)", "resnet50_gem", lambda x: gem_pool(body(x)))]:
        emb[name], t = extract(key, fn, prep224, paths)
        cost[name] = (n_params(m), t)


def vit_embeddings(paths, emb, cost):
    m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).eval().to(device)

    def tokens(x):
        x = m._process_input(x)                              # image -> patch tokens
        x = torch.cat([m.class_token.expand(x.shape[0], -1, -1), x], 1)
        return m.encoder(x)

    for name, key, fn in [("ViT-B/16 (CLS)", "vit_cls", lambda x: tokens(x)[:, 0]),
                          ("ViT-B/16 (mean patch)", "vit_mean", lambda x: tokens(x)[:, 1:].mean(1))]:
        emb[name], t = extract(key, fn, prep224, paths)
        cost[name] = (n_params(m), t)


def pretrained_embeddings(paths):
    emb, cost = {}, {}
    for fn in (resnet_embeddings, vit_embeddings):
        fn(paths, emb, cost)
        torch.cuda.empty_cache()
    return emb, cost
