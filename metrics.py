import numpy as np
from config import K, SEED

_rng = np.random.default_rng(SEED)


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def retrieve(emb, q_idx, k=K, metric="cosine"):
    # top-k neighbours from the whole catalogue, query itself excluded
    q_idx = np.asarray(q_idx)
    if metric == "cosine":
        e = l2norm(emb.astype(np.float32))
        sim = e[q_idx] @ e.T
    else:
        e = emb.astype(np.float32)
        q = e[q_idx]
        sim = -((q ** 2).sum(1)[:, None] - 2 * q @ e.T + (e ** 2).sum(1)[None, :])
    sim[np.arange(len(q_idx)), q_idx] = -np.inf
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, top, 1), axis=1)
    return np.take_along_axis(top, order, 1)


def per_query(nbrs, y, q_idx, k=K):
    hits = y[nbrs] == y[q_idx][:, None]
    n_rel = np.bincount(y)[y[q_idx]] - 1
    ok = n_rel > 0                                   # skip queries with no other relevant item
    prec = hits.mean(1)
    ap = ((np.cumsum(hits, 1) / np.arange(1, k + 1)) * hits).sum(1) / np.minimum(k, np.maximum(n_rel, 1))
    rec = hits.sum(1) / np.maximum(n_rel, 1)
    return prec[ok], ap[ok], rec[ok]


def bootstrap_ci(v, n=1000):
    means = _rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def evaluate(emb, y, q_idx, metric="cosine", k=K):
    nbrs = retrieve(emb, q_idx, k, metric)
    p, ap, r = per_query(nbrs, y, q_idx, k)
    lo, hi = bootstrap_ci(ap)
    return {f"P@{k}": p.mean(), f"MAP@{k}": ap.mean(), f"R@{k}": r.mean(), "MAP_lo": lo, "MAP_hi": hi}


def paired_diff(emb_a, emb_b, y, q_idx, k=K, n=1000):
    # same queries for both models, so bootstrap the per-query difference in AP
    ap_a = per_query(retrieve(emb_a, q_idx, k), y, q_idx, k)[1]
    ap_b = per_query(retrieve(emb_b, q_idx, k), y, q_idx, k)[1]
    d = ap_a - ap_b
    means = _rng.choice(d, (n, len(d))).mean(1)
    return d.mean(), np.percentile(means, [2.5, 97.5])
