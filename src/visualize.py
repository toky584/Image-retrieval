import matplotlib.pyplot as plt
from typing import List
from PIL import Image

def plot_retrieval_results(query_img: Image.Image, retrieved_imgs: List[Image.Image], k: int = 5):
    """Plots the query image alongside the top k retrieved images."""
    plt.figure(figsize=(12, 4))
    
    # Plot query
    plt.subplot(1, k + 1, 1)
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis('off')
    
    # Plot retrieved
    for i in range(min(k, len(retrieved_imgs))):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(retrieved_imgs[i])
        plt.title(f"Rank {i + 1}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
