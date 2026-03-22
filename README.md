# Image Retrieval for Vehicle Re-Identification

This repository provides an image retrieval system designed for vehicle re-identification, utilizing the VeRi dataset. The core objective is to learn robust embedding representations for vehicle images so that identical vehicles are clustered closer together in the embedding space, while different vehicles are pushed apart.

## Overview

The system employs a pre-trained ResNet-50 model adapted for feature extraction. The final fully connected layer is replaced with a custom embedding layer, and the network is fine-tuned using a **Triplet Margin Loss**. This approach enforces similarity among matching vehicles (positive pairs) and penalizes similarity among different vehicles (negative pairs).

## Features

- **Data Processing (`src/data_processing.py`)**: Custom PyTorch `Dataset` that dynamically generates triplets (anchor, positive, negative) for efficient training.
- **Model Architecture (`src/model.py`)**: Customized ResNet-50 backbone with a 416-dimensional embedding space.
- **Training Loop (`src/train.py`)**: Dedicated training script that implements Triplet Loss optimization.
- **Evaluation (`src/evaluate.py`)**: Evaluation utilities for calculating Euclidean distance, retrieving top-k matches, and computing Mean Average Precision (mAP).
- **Visualization (`src/visualize.py`)**: Tools for visually inspecting query images against their retrieved rank-k results.

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision numpy Pillow matplotlib scipy
```

## Usage

*Note: The actual dataset (VeRi) should be placed in your working directory under appropriate folders (e.g., `image_query`, `image_test`, `image_train`).*

### 1. Training

Initialize the model and pass your dataloader to the training routine:

```python
import torch
from torch.utils.data import DataLoader
from src.data_processing import load_images_and_labels, TripletDataset, get_transform
from src.model import EmbeddingModel
from src.train import train_model

# 1. Load Data
images, labels = load_images_and_labels('image_train')
dataset = TripletDataset(images, labels, transform=get_transform())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Initialize Model
model = EmbeddingModel(pretrained=True)

# 3. Train
train_model(model, dataloader, num_epochs=10, save_path="embedding_model.pth")
```

### 2. Evaluation & Visualization

Use the evaluation tools to compute embeddings and assess precision:

```python
from src.evaluate import retrieve_top_k, calculate_map, create_relevance_matrix
from src.visualize import plot_retrieval_results

# Example embeddings and distance calculation
# top_k_indices = retrieve_top_k(query_embeddings, test_embeddings, k=5)

# Calculate mAP
# relevance = create_relevance_matrix(query_labels, test_labels)
# m_ap = calculate_map(top_k_indices, relevance)
# print(f"Mean Average Precision: {m_ap}")

# Visualize
# plot_retrieval_results(query_images[0], [test_images[i] for i in top_k_indices[0]])
```

## Project Structure

```text
├── README.md
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── evaluate.py
    ├── model.py
    ├── train.py
    └── visualize.py
```

## License

This project is open-source and available under the MIT License.
