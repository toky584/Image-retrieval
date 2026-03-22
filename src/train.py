import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    num_epochs: int = 10, 
    lr: float = 0.001,
    device: str = "cpu",
    save_path: str = "embedding_model.pth"
):
    """
    Trains the embedding model using Triplet Margin Loss.
    
    Args:
        model: The PyTorch embedding model.
        dataloader: DataLoader returning (anchors, positives, negatives).
        num_epochs: Number of epochs to train.
        lr: Learning rate for Adam optimizer.
        device: 'cpu' or 'cuda'.
        save_path: Path to save the best/last model.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out_anch = model(anchors)
            out_pos = model(positives)
            out_neg = model(negatives)
            
            # Compute loss
            loss = F.triplet_margin_loss(out_anch, out_pos, out_neg, margin=1.0, p=2)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
