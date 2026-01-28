"""
Training script for the obfuscation readability detection model.

This module implements the training loop for the CharCNN model. It loads the dataset,
initializes the model and optimizer, and trains the model using binary cross-entropy loss.
The trained model is saved as a PyTorch state dictionary for later use.
"""

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LR
from charset import vocab_size
from dataset import NameDataset
from model import CharCNN


def main():
    """
    Main training function that orchestrates the model training process.
    
    Process:
        1. Load and prepare the training dataset
        2. Initialize the CharCNN model
        3. Set up optimizer and loss function
        4. Train the model for specified epochs
        5. Save the trained model weights
    """
    
    # Load training dataset
    # Combines readable and obfuscated identifiers with their labels
    dataset = NameDataset(
        "data/readable.txt",      # File containing readable identifiers (label=1)
        "data/obfuscated.txt"     # File containing obfuscated identifiers (label=0)
    )

    # Create data loader for batch processing
    # shuffle=True ensures random order during training
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,    # Number of samples per batch
        shuffle=True              # Randomize sample order
    )

    # Initialize the CharCNN model
    model = CharCNN(vocab_size)
    
    # Set up Adam optimizer with learning rate from config
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Binary Cross-Entropy loss for binary classification
    loss_fn = torch.nn.BCELoss()

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LR}")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        # Iterate through batches
        for x, y in loader:
            # Clear gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            pred = model(x)
            
            # Compute loss between predictions and true labels
            loss = loss_fn(pred, y)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Accumulate loss for logging
            total_loss += loss.item()

        # Print epoch progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS}, loss={total_loss:.4f}")

    # Save trained model weights
    torch.save(model.state_dict(), "models/readability.pt")
    print(f"\n✅ Training completed! Model saved to models/readability.pt")


if __name__ == "__main__":
    main()
