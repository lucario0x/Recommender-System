# Visual Fashion Recommender System

An image-based product recommender for fashion items. Given a product image, the system retrieves the most visually similar items using learned image embeddings and nearest-neighbour search. Two embedding approaches are compared against a random baseline:

- **Convolutional Autoencoder** trained from scratch
- **ResNet-50** pretrained on ImageNet (used as a frozen feature extractor)

## Overview

1. Load the fashion product catalogue (`styles.csv` + product images).
2. Filter out rare article types and sample a subset of products.
3. Extract embeddings with each method.
4. Retrieve the top-K most similar items using cosine similarity.
5. Evaluate with **Precision@K** and **MAP@K** on a held-out test set, treating items of the same `articleType` as relevant.
6. Visualise example recommendations.

## Dataset

Fashion Product Images dataset (product metadata in `styles.csv` and images in `images/`). Place the downloaded `archive.zip` in the working directory (the notebook expects `/content/archive.zip` on Google Colab); it is extracted automatically.

Preprocessing:

| Setting | Value |
|---|---|
| Products used (`N_ITEMS`) | 5,000 |
| Minimum items per article type (`MIN_PER_TYPE`) | 30 |
| Train / test split | 80 / 20 |
| Random seed | 0 |

## Methods

### 1. Random baseline
Random 64-dimensional vectors, used as a lower bound.

### 2. Autoencoder (from scratch)
- Input: images resized to 64×64
- Encoder: 3 strided Conv layers (32 → 64 → 128 channels) followed by a linear layer to a **128-d latent vector**
- Decoder: linear layer and 3 transposed Conv layers with a Sigmoid output
- Loss: MSE reconstruction loss, Adam optimiser (lr = 1e-3), 20 epochs, batch size 128
- The 128-d latent vector is used as the item embedding.

### 3. ResNet-50 (pretrained)
- Torchvision `ResNet50` with `IMAGENET1K_V2` weights
- Final classification layer removed, giving a **2048-d** feature vector
- Images resized to 224×224 and normalised with ImageNet statistics

## Evaluation

For each test item, the top-K (K = 10) nearest neighbours are retrieved using cosine distance, excluding the query itself. A neighbour counts as relevant if it has the same `articleType` as the query.

- **Precision@K**: fraction of the K recommendations that are relevant
- **MAP@K**: mean average precision at K, which also rewards placing relevant items earlier in the ranking

## Results

Results on the held-out test set (also saved to `results.csv`):

| Method | Precision@10 | MAP@10 |
|---|---|---|
| Random | _fill in_ | _fill in_ |
| Autoencoder (from scratch) | _fill in_ | _fill in_ |
| ResNet-50 (pretrained) | _fill in_ | _fill in_ |

![MAP comparison](map_comparison.png)

Example recommendations (first column is the query):

![Example recommendations](examples.png)

## Getting Started

### Requirements

- Python 3.8+
- PyTorch and torchvision
- NumPy, pandas, scikit-learn
- Matplotlib, Pillow, tqdm

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow tqdm
```

A GPU is recommended but not required (the notebook picks CUDA automatically if available).

### Run

1. Open `Recommender_System.ipynb` in Google Colab or Jupyter.
2. Upload the dataset archive and set `ZIP_PATH` in the first cell if needed.
3. Run all cells.

## Configuration

Key parameters at the top of the notebook:

```python
N_ITEMS      = 5000   # number of products used (more = better, slower)
MIN_PER_TYPE = 30     # drop very rare article types
AE_EPOCHS    = 20     # autoencoder training epochs
K            = 10     # number of recommendations
SEED         = 0
```

## Outputs

- `results.csv`: Precision@K and MAP@K for each method
- `map_comparison.png`: bar chart of MAP@K
- `examples.png`: qualitative recommendation examples

## Limitations and Future Work

- Relevance is defined by matching `articleType`, so visually similar items of different types (e.g. colour or style matches) count as misses.
- Only image content is used; metadata such as colour, gender, season or brand could be combined with the embeddings.
- Possible improvements: fine-tuning ResNet, trying CLIP or ViT embeddings, using approximate nearest-neighbour search (FAISS) for larger catalogues, and adding user interaction data for collaborative filtering.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib
