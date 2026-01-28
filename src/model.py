"""
Character-level Convolutional Neural Network for identifier readability detection.

This module implements a CharCNN model that processes identifier names character by character
to classify them as readable or obfuscated. The model uses character embeddings, 1D convolution,
and global average pooling to learn patterns in identifier names.
"""

import torch
import torch.nn as nn
from config import EMBED_DIM, NUM_FILTERS, KERNEL_SIZE


class CharCNN(nn.Module):
    """
    Character-level Convolutional Neural Network for identifier classification.
    
    Architecture:
        1. Character Embedding Layer: Converts character indices to dense vectors
        2. 1D Convolution Layer: Extracts local character patterns
        3. Global Average Pooling: Aggregates features across sequence length
        4. Fully Connected Layer: Produces binary classification output
    
    The model outputs a probability score between 0 and 1, where higher values
    indicate more readable identifiers.
    """
    
    def __init__(self, vocab_size):
        """
        Initialize the CharCNN model.
        
        Args:
            vocab_size (int): Size of the character vocabulary (includes padding token)
        """
        super().__init__()

        # Character embedding layer
        # Converts character indices to dense embedding vectors
        # padding_idx=0 ensures padding tokens get zero vectors
        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,           # Embedding dimension
            padding_idx=0        # Padding token index
        )

        # 1D Convolution layer
        # Extracts local character patterns using sliding windows
        self.conv = nn.Conv1d(
            EMBED_DIM,           # Input channels (embedding dimension)
            NUM_FILTERS,         # Output channels (number of filters)
            kernel_size=KERNEL_SIZE  # Filter size
        )

        # Global Average Pooling
        # Reduces sequence dimension to single value per filter
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        # Maps pooled features to binary classification output
        self.fc = nn.Linear(NUM_FILTERS, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length)
                       containing character indices
        
        Returns:
            Tensor: Output tensor of shape (batch_size, 1) with probability scores
        
        Shape transformations:
            Input:     [B, L]           - Batch of character sequences
            Embedding: [B, L, E]        - Character embeddings
            Conv:      [B, F, L']       - Convolutional features
            Pool:      [B, F, 1]        - Pooled features
            FC:        [B, 1]           - Binary classification scores
            Sigmoid:   [B, 1]           - Probability scores [0, 1]
            
        Where:
            B = batch_size
            L = sequence_length (MAX_LEN)
            E = embedding_dimension (EMBED_DIM)
            F = num_filters (NUM_FILTERS)
            L' = output_length after convolution
        """
        # Convert character indices to embeddings
        x = self.embedding(x)        # [B, L, E]
        
        # Transpose for convolution: (batch, channels, length)
        x = x.transpose(1, 2)        # [B, E, L]
        
        # Apply convolution and ReLU activation
        x = torch.relu(self.conv(x)) # [B, F, L']
        
        # Global average pooling across sequence dimension
        x = self.pool(x).squeeze(-1) # [B, F]
        
        # Linear transformation to single output
        x = torch.sigmoid(self.fc(x)) # [B, 1] - probabilities
        
        return x
