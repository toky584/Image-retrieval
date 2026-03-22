import torch
import numpy as np
from typing import List
from scipy.spatial import distance

def extract_embeddings(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cpu") -> np.ndarray:
    """Extracts embeddings for all images in the dataloader."""
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            out = model(images)
            embeddings.append(out.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def retrieve_top_k(query_embeddings: np.ndarray, test_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Returns the indices of the top k closest test images for each query."""
    distances = distance.cdist(query_embeddings, test_embeddings, metric='euclidean')
    top_k_indices = np.argsort(distances, axis=1)[:, :k]
    return top_k_indices

def create_relevance_matrix(query_labels: List[str], test_labels: List[str]) -> np.ndarray:
    """Creates a boolean matrix indicating if query and test labels match."""
    q_labels = np.array(query_labels)
    t_labels = np.array(test_labels)
    return (q_labels[:, None] == t_labels[None, :]).astype(int)

def calculate_precision_for_kth(top_indices: np.ndarray, relevant_indices: set) -> List[float]:
    """Calculates precision at each rank for a single query."""
    precisions = []
    relevant = 0
    for i, index in enumerate(top_indices):
        if index in relevant_indices:
            relevant += 1
            precisions.append(relevant / (i + 1))
    return precisions if precisions else [0.0]

def calculate_map(top_k_indices: np.ndarray, relevance_matrix: np.ndarray) -> float:
    """Calculates Mean Average Precision (mAP)."""
    ap_list = []
    for query_idx in range(relevance_matrix.shape[0]):
        top_indices = top_k_indices[query_idx]
        relevant_indices = set(np.where(relevance_matrix[query_idx] == 1)[0])
        precisions = calculate_precision_for_kth(top_indices, relevant_indices)
        ap_list.append(np.mean(precisions))
    return float(np.mean(ap_list))
