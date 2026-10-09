import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import device, SEED, VIT_EPOCHS
from features import n_params
from scratch import embed_batches


class PatchEmbed(nn.Module):
    # a conv with kernel = stride = patch size cuts the image into patches and projects each one
    def __init__(self, img, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.n = (img // patch) ** 2

    def forward(self, x):                                   # (B,3,H,W) -> (B,N,dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / self.hd ** 0.5   # how much each token looks at every other token
        attn = attn.softmax(dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))                       # residual connections
        return x + self.mlp(self.n2(x))


class ViTAutoEncoder(nn.Module):
    # transformer encoder -> one vector -> linear decoder that rebuilds every patch
    def __init__(self, img=64, patch=8, dim=128, depth=4, heads=4, pool="cls", pos="learned", latent=128):
        super().__init__()
        self.patch, self.pool = patch, pool
        self.embed = PatchEmbed(img, patch, dim)
        n = self.embed.n
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        n_tok = n + (1 if pool == "cls" else 0)
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02) if pos == "learned" else None
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Linear(dim, latent)
        self.decoder = nn.Linear(latent, n * 3 * patch * patch)

    def encode(self, x):
        t = self.embed(x)
        if self.cls is not None:
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        if self.pos is not None:
            t = t + self.pos
        t = self.norm(self.blocks(t))
        z = t[:, 0] if self.pool == "cls" else t.mean(1)    # cls token vs average of patch tokens
        return self.to_latent(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z).view(x.shape[0], self.embed.n, -1)
        target = F.unfold(x, self.patch, stride=self.patch).transpose(1, 2)   # image as patches
        return recon, target, z

    def encoder_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("decoder")) / 1e6


def train_vit(small, train_idx, patch=8, pool="cls", pos="learned", epochs=VIT_EPOCHS, bs=128, name=""):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = ViTAutoEncoder(img=small.shape[-1], patch=patch, pool=pool, pos=pos).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total = 0
        for i in range(0, len(perm), bs):
            x = small[perm[i:i + bs]].to(device)
            flip = torch.rand(x.shape[0], 1, 1, 1, device=device) < 0.5
            x = torch.where(flip, x.flip(3), x)             # random horizontal flip
            recon, target, _ = model(x)
            loss = F.mse_loss(recon, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(x)
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"vit {name} epoch {ep + 1}: recon {total / len(perm):.4f}")
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()
    emb = embed_batches(model.encode, small)
    return emb, (model.encoder_params(), secs)
