# Visual Fashion Recommender System

An image-based product recommender for fashion items. Given a product image, the system retrieves the most visually similar items using image embeddings and cosine nearest-neighbour search. Four embedding models are benchmarked against a random baseline, with ablations on pooling, dimensionality and distance metric.

## Models Compared

| Model | Type | Embedding dim |
|---|---|---|
| **ViT-B/16** (CLS token, mean-patch pooling) | Pretrained, frozen | 768 |
| **ResNet-50** (average pooling, GeM pooling) | Pretrained, frozen | 2048 |
| **CNN Autoencoder** | Trained from scratch | 128 |
| **ViT** (small, patch size 8) | Trained from scratch | 128 |
| Random vectors | Lower-bound baseline | 64 |

## Overview

1. Load the fashion product catalogue (`styles.csv` + product images).
2. Filter rare article types and sample a subset of products (4,976 products, 75 article types).
3. Split into train (3,980) and query (996) sets; the from-scratch models train on the train split only.
4. Extract embeddings with each method.
5. Retrieve the top-K (K = 10) most similar items using cosine similarity, excluding the query itself.
6. Evaluate with Precision@10, MAP@10 and Recall@10, with bootstrap 95% confidence intervals and paired significance tests.
7. Run ablations, error analysis and a cost comparison, then save the best index for serving.

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle.

Dataset mirror: https://drive.google.com/file/d/129OC_SRlOpe-O1Bqa2CJsnQbzMYhQzhe/view?usp=sharing

| Setting | Value |
|---|---|
| Products used | 4,976 |
| Article types | 75 |
| Train / query split | 3,980 / 996 |
| Random seed | 0 |

## Evaluation

A retrieved neighbour counts as relevant if it matches the query's label. The main definition is **same `articleType`**. Two other definitions are reported to show how results change with strictness:

- `masterCategory` (coarse)
- `articleType` (main)
- `articleType + baseColour` (strict)

Metrics:

- **Precision@K**: fraction of the K recommendations that are relevant
- **MAP@K**: mean average precision at K, which also rewards placing relevant items earlier
- **Recall@K**: fraction of all relevant items retrieved in the top K (low in absolute terms because many items share each type)

## Results

Main comparison (article-type relevance, K = 10):

| Method | Dim | Precision@10 | MAP@10 | MAP@10 95% CI | Recall@10 |
|---|---|---|---|---|---|
| **ViT-B/16 (CLS)** | 768 | **0.685** | **0.620** | 0.598 – 0.644 | 0.070 |
| ResNet-50 (GeM pool) | 2048 | 0.682 | 0.614 | 0.593 – 0.637 | 0.073 |
| ResNet-50 (avg pool) | 2048 | 0.676 | 0.605 | 0.582 – 0.629 | 0.071 |
| ViT-B/16 (mean patch) | 768 | 0.657 | 0.589 | 0.567 – 0.612 | 0.068 |
| CNN Autoencoder (scratch) | 128 | 0.512 | 0.427 | 0.405 – 0.448 | 0.049 |
| ViT (scratch) | 128 | 0.190 | 0.117 | 0.107 – 0.128 | 0.015 |
| Random | 64 | 0.048 | 0.018 | 0.015 – 0.021 | 0.002 |

Paired MAP@10 differences (bootstrap 95% CI):

| Comparison | Difference |
|---|---|
| ViT-B/16 (CLS) vs ResNet-50 (avg pool) | +0.015 (+0.002 to +0.027) |
| ViT-B/16 (CLS) vs ViT-B/16 (mean patch) | +0.031 (+0.018 to +0.044) |
| ResNet-50 (GeM) vs ResNet-50 (avg pool) | +0.009 (+0.001 to +0.016) |

### Ablations

**Distance metric (ResNet-50, avg pool):** cosine 0.606 MAP@10 vs Euclidean 0.596.

**PCA dimensionality (ViT-B/16 CLS):** quality saturates around 128 dimensions.

| Dim | MAP@10 | P@10 | Index size (MB) |
|---|---|---|---|
| 16 | 0.558 | 0.635 | 0.32 |
| 32 | 0.602 | 0.672 | 0.64 |
| 64 | 0.618 | 0.684 | 1.27 |
| 128 | 0.623 | 0.688 | 2.55 |
| 256 | 0.624 | 0.689 | 5.10 |
| 512 | 0.624 | 0.689 | 10.19 |
| 768 | 0.620 | 0.685 | 15.29 |

**Scratch ViT design (MAP@10):**

| Config | Tokens | MAP@10 | P@10 |
|---|---|---|---|
| patch 8, CLS, learned pos. | 64 | 0.117 | 0.190 |
| patch 8, mean pool, learned pos. | 64 | 0.141 | 0.219 |
| patch 4, CLS, learned pos. | 256 | 0.099 | 0.171 |
| patch 16, CLS, learned pos. | 16 | 0.135 | 0.211 |
| patch 8, CLS, no pos. | 64 | 0.120 | 0.197 |

### Relevance strictness (MAP@10)

| Method | masterCategory | articleType | articleType + colour |
|---|---|---|---|
| ViT-B/16 (CLS) | 0.974 | 0.620 | 0.181 |
| ResNet-50 (avg pool) | 0.973 | 0.605 | 0.171 |
| CNN Autoencoder (scratch) | 0.894 | 0.427 | 0.113 |
| ViT (scratch) | 0.522 | 0.117 | 0.047 |

### Cost

| Method | Params (M) | Dim | Embedding time (s) | Index (MB) | Query (ms) | MAP@10 |
|---|---|---|---|---|---|---|
| ViT-B/16 (CLS) | 86.6 | 768 | 16.3 | 15.3 | 0.185 | 0.620 |
| ResNet-50 (GeM pool) | 25.6 | 2048 | 12.9 | 40.8 | 0.275 | 0.614 |
| ResNet-50 (avg pool) | 25.6 | 2048 | 13.1 | 40.8 | 0.275 | 0.605 |
| ViT-B/16 (mean patch) | 86.6 | 768 | 16.8 | 15.3 | 0.191 | 0.589 |
| CNN Autoencoder (scratch) | 0.5 | 128 | 5.9 | 2.5 | 0.125 | 0.427 |
| ViT (scratch) | 0.8 | 128 | 4.5 | 2.5 | 0.087 | 0.117 |

### Error analysis

Hardest article types for ViT-B/16 (mean per-class score): Clutches (0.19), Flats (0.20), Dresses (0.24), Tops (0.40), Sandals (0.45).

Most common confusions are between visually near-identical categories: Casual Shoes ↔ Formal/Sports Shoes, Tshirts ↔ Tops, Tops ↔ Kurtas, and Heels ↔ Flats. `examples.png` and `failures.png` show sample successes and the worst queries.

## Key Takeaways

- Pretrained embeddings clearly beat models trained from scratch on ~4k images.
- ViT-B/16 edges out ResNet-50 by a small but statistically significant margin, with a much smaller embedding (768-d vs 2048-d).
- CLS-token pooling beats mean-patch pooling for ViT-B/16; GeM pooling beats average pooling for ResNet-50.
- PCA to 128 dimensions matches full-dimension quality while cutting the index about 6×.
- Cosine distance is slightly better than Euclidean.
- A ViT trained from scratch on a few thousand images performs poorly, as expected without large-scale data.

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Paths, seed, K and other settings |
| `data.py` | Catalogue loading, filtering and train/query split |
| `features.py` | Pretrained ResNet-50 and ViT-B/16 embeddings |
| `scratch.py` | CNN autoencoder and ViT trained from scratch |
| `analysis.py` | Metrics, confidence intervals, ablations, error analysis, plots |
| `serve.py` | Saves the best embedding index for retrieval |
| `Recommendation_System.ipynb` | Notebook that runs the full pipeline |

## Getting Started

### Requirements

- Python 3.8+
- PyTorch and torchvision
- NumPy, pandas, scikit-learn
- Matplotlib, Pillow, tqdm
- kagglehub (to download the dataset)

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow tqdm kagglehub
```

A GPU is recommended but not required.

### Run

1. Open `Recommendation_System.ipynb` in Google Colab or Jupyter.
2. Upload `fashion_recsys.zip` (the project modules) and run the first cell to unpack it.
3. Run all cells; the dataset is downloaded automatically through `kagglehub`.

## Limitations and Future Work

- Relevance is defined by matching `articleType`, so visually similar items of different types (e.g. colour or style matches) count as misses. The strict colour-aware metric shows scores drop sharply under it.
- Only image content is used; metadata such as colour, gender, season or brand could be combined with the embeddings.
- Possible improvements: fine-tuning the pretrained backbones, trying CLIP for text-to-image search, approximate nearest-neighbour search (FAISS) for larger catalogues, and adding user interaction data for collaborative filtering.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib
