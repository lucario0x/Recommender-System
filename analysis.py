import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

import vit_scratch
from config import K, SEED, OUT, device
from metrics import evaluate, retrieve, l2norm, paired_diff

try:
    from IPython.display import display
except ImportError:
    display = print

# one change at a time, starting from the first row
VIT_CONFIGS = [
    dict(patch=8, pool="cls", pos="learned"),
    dict(patch=8, pool="mean", pos="learned"),
    dict(patch=4, pool="cls", pos="learned"),
    dict(patch=16, pool="cls", pos="learned"),
    dict(patch=8, pool="cls", pos="none"),
]


def compare(emb, cat):
    rows = []
    for name, e in emb.items():
        r = evaluate(e, cat.labels, cat.test_idx)
        r.update(Method=name, Dim=e.shape[1])
        rows.append(r)
    res = pd.DataFrame(rows).set_index("Method").sort_values(f"MAP@{K}", ascending=False)
    res.round(4).to_csv(f"{OUT}/results.csv")
    best = res.drop(index="Random").index[0]
    return res, best


def paired(emb, cat, a, b):
    diff, (lo, hi) = paired_diff(emb[a], emb[b], cat.labels, cat.test_idx)
    print(f"{a} - {b}: {diff:+.3f} MAP@{K}  (95% CI {lo:+.3f} to {hi:+.3f})")


def plot_comparison(res):
    r = res.iloc[::-1]
    m = r[f"MAP@{K}"]
    plt.figure(figsize=(8, 4.5))
    plt.barh(r.index, m, xerr=[m - r["MAP_lo"], r["MAP_hi"] - m], color="steelblue", capsize=3)
    plt.xlabel(f"MAP@{K} (95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/map_comparison.png", dpi=120)
    plt.show()


def vit_ablation(small, cat, configs=VIT_CONFIGS):
    rows, embs, costs = [], {}, {}
    img = small.shape[-1]
    for c in configs:
        label = f"p{c['patch']} {c['pool']} {c['pos']}"
        e, cst = vit_scratch.train_vit(small, cat.train_idx, name=label, **c)
        r = evaluate(e, cat.labels, cat.test_idx)
        rows.append({"Config": label, "Patch": c["patch"], "Pool": c["pool"], "Pos": c["pos"],
                     "Tokens": (img // c["patch"]) ** 2, f"MAP@{K}": r[f"MAP@{K}"], f"P@{K}": r[f"P@{K}"],
                     "Params (M)": cst[0], "Train (s)": cst[1]})
        embs[label], costs[label] = e, cst
    out = pd.DataFrame(rows).set_index("Config")
    out.round(3).to_csv(f"{OUT}/vit_ablation.csv")
    return out, embs, costs


def pca_ablation(base, cat, dims=(16, 32, 64, 128, 256, 512)):
    rows = []
    full = base.shape[1]
    for d in [*[d for d in dims if d < min(full, len(cat.train_idx))], full]:
        # pca fitted on train items only
        e = base if d == full else PCA(d, random_state=SEED).fit(base[cat.train_idx]).transform(base)
        r = evaluate(e, cat.labels, cat.test_idx)
        rows.append({"Dim": d, f"MAP@{K}": r[f"MAP@{K}"], f"P@{K}": r[f"P@{K}"], "Index MB": len(e) * d * 4 / 1e6})
    out = pd.DataFrame(rows)
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(out["Dim"], out[f"MAP@{K}"], "o-")
    plt.xscale("log", base=2)
    plt.xlabel("embedding dim (PCA)")
    plt.ylabel(f"MAP@{K}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/pca_ablation.png", dpi=120)
    plt.show()
    return out


def metric_ablation(e, cat):
    return {m: round(float(evaluate(e, cat.labels, cat.test_idx, metric=m)[f"MAP@{K}"]), 4)
            for m in ("cosine", "euclidean")}


def relevance_table(emb, names, cat):
    rows = [{"Method": n, "Relevance": ln, f"MAP@{K}": evaluate(emb[n], y, cat.test_idx)[f"MAP@{K}"]}
            for n in names if n in emb for ln, y in cat.label_sets.items()]
    return pd.DataFrame(rows).pivot(index="Method", columns="Relevance", values=f"MAP@{K}")[list(cat.label_sets)]


def query_latency_ms(e, n=200):
    t = torch.tensor(l2norm(e.astype(np.float32)), device=device)
    qs = t[torch.randint(0, len(t), (n,))]
    for q in qs[:10]:                       # warm up
        (t @ q).topk(K)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for q in qs:
        (t @ q).topk(K)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def cost_table(emb, cost, res):
    # time = embedding time for pretrained models, training time for scratch models
    rows = []
    for name, e in emb.items():
        if name == "Random":
            continue
        p, t = cost.get(name, (np.nan, None))
        rows.append({"Method": name, "Params (M)": p, "Dim": e.shape[1], "Time (s)": t,
                     "Index (MB)": e.size * 4 / 1e6, "Query (ms)": query_latency_ms(e),
                     f"MAP@{K}": res.loc[name, f"MAP@{K}"]})
    out = pd.DataFrame(rows).set_index("Method").sort_values(f"MAP@{K}", ascending=False)
    out.round(3).to_csv(f"{OUT}/cost_vs_accuracy.csv")

    plt.figure(figsize=(6.5, 4.5))
    for name, r in out.iterrows():
        plt.scatter(r["Index (MB)"], r[f"MAP@{K}"], s=60)
        plt.annotate(name, (r["Index (MB)"], r[f"MAP@{K}"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("index size (MB)")
    plt.ylabel(f"MAP@{K}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/accuracy_vs_cost.png", dpi=120)
    plt.show()
    return out


def error_analysis(e, cat):
    q = cat.test_idx
    nbrs = retrieve(e, q)
    hits = cat.labels[nbrs] == cat.labels[q][:, None]
    per_class = (pd.DataFrame({"type": cat.type_names[cat.labels[q]], f"P@{K}": hits.mean(1)})
                 .groupby("type")[f"P@{K}"].agg(["mean", "count"]))
    worst = per_class[per_class["count"] >= 8].sort_values("mean").head(10)

    top1 = nbrs[:, 0]
    bad = cat.labels[top1] != cat.labels[q]
    pairs = pd.Series([f"{cat.type_names[cat.labels[a]]} -> {cat.type_names[cat.labels[b]]}"
                       for a, b in zip(q[bad], top1[bad])]).value_counts().head(10)
    worst_queries = q[np.argsort(hits.mean(1))[:4]]
    return worst.round(3), pairs, worst_queries


def show(e, cat, queries, k=5, fname="examples.png"):
    queries = np.asarray(queries)
    nb = retrieve(e, queries, k)
    fig, ax = plt.subplots(len(queries), k + 1, figsize=(2.1 * (k + 1), 2.4 * len(queries)))
    for r, (qi, row) in enumerate(zip(queries, nb)):
        for c, j in enumerate([qi, *row]):
            ax[r, c].imshow(Image.open(cat.paths[j]))
            ax[r, c].axis("off")
            same = cat.labels[j] == cat.labels[qi]
            ax[r, c].set_title("query" if c == 0 else cat.df["articleType"][j], fontsize=8,
                               color="black" if c == 0 else ("green" if same else "red"))
    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=110)
    plt.show()
