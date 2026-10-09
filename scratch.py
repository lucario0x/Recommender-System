import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import device, SEED, AE_EPOCHS
from features import n_params


def embed_batches(fn, small, bs=256):
    # run fn over all images in chunks so big models don't run out of memory
    out = []
    with torch.no_grad():
        for i in range(0, len(small), bs):
            out.append(fn(small[i:i + bs].to(device)).cpu())
    return torch.cat(out).numpy()


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class AutoEncoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
                                     conv_block(128, 128), nn.Flatten(), nn.Linear(128 * 4 * 4, latent))
        self.fc = nn.Linear(latent, 128 * 4 * 4)
        up = lambda i, o: nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.decoder = nn.Sequential(up(128, 128), up(128, 64), up(64, 32),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.fc(z).view(-1, 128, 4, 4)), z


def train_autoencoder(small, train_idx, epochs=AE_EPOCHS):
    rng = np.random.default_rng(SEED)
    ae = AutoEncoder().to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(epochs):
        ae.train()
        perm = rng.permutation(train_idx)
        total = 0
        for i in range(0, len(perm), 128):
            x = small[perm[i:i + 128]].to(device)
            loss = F.mse_loss(ae(x)[0], x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(x)
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"ae epoch {ep + 1}: recon {total / len(perm):.4f}")
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    ae.eval()
    emb = embed_batches(lambda x: ae(x)[1], small)
    return emb, (n_params(ae.encoder), secs)
