# Visual Fashion Recommender

An image-based fashion recommender. Given a product image, it finds the most visually similar products using image embeddings and cosine nearest-neighbour search.

I compared four embedding models against a random baseline:

- **ResNet-50** (pretrained on ImageNet, frozen)
- **ViT-B/16** (pretrained on ImageNet, frozen)
- **CNN autoencoder** (trained from scratch)
- **ViT** (small, trained from scratch)

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small): 4,976 products, 75 article types, split 3,980 train / 996 query.

## Evaluation

For each query image, the top 10 nearest neighbours are retrieved (excluding the query). A neighbour is relevant if it has the same `articleType`. I report **Precision@10** and **MAP@10**.

## Results

| Method | Precision@10 | MAP@10 |
|---|---|---|
| ViT-B/16 (pretrained) | 0.685 | 0.620 |
| ResNet-50 (pretrained) | 0.676 | 0.605 |
| CNN Autoencoder (scratch) | 0.512 | 0.427 |
| ViT (scratch) | 0.190 | 0.117 |
| Random | 0.048 | 0.018 |

## Findings

- Pretrained embeddings are much better than models trained from scratch on ~4k images.
- ViT-B/16 is only slightly better than ResNet-50, but its embedding is smaller (768-d vs 2048-d).
- Reducing the ViT-B/16 embeddings to 128 dimensions with PCA keeps the same quality (MAP@10 0.623 vs 0.620) and makes the index about 6x smaller.
- Most mistakes are between similar categories, such as casual vs formal shoes, or tops vs t-shirts.

## Run

1. Open `Recommendation_System.ipynb` in Google Colab.
2. Upload `fashion_recsys.zip` and run all cells. The dataset downloads automatically.

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow tqdm kagglehub
```

## Limitations

- "Relevant" means same article type, so visually similar items of a different type count as wrong.
- Only image content is used; colour, brand and other metadata are ignored.

## Future Work

Fine-tuning the backbones, CLIP for text search, FAISS for larger catalogues.
