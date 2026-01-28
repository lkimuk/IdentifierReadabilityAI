"""
Dataset class for training the obfuscation readability detection model.

This module defines a PyTorch Dataset that loads identifier names from text files
and prepares them for training. It handles both readable and obfuscated identifiers
with corresponding labels for binary classification.
"""

import torch
from torch.utils.data import Dataset
from charset import encode


class NameDataset(Dataset):
    """
    PyTorch Dataset for identifier readability classification.
    
    This dataset loads identifier names from two text files:
    - readable_path: Contains readable identifier names (label=1)
    - obfuscated_path: Contains obfuscated identifier names (label=0)
    
    Each identifier is encoded into a numerical sequence using the charset module.
    """
    
    def __init__(self, readable_path, obfuscated_path):
        """
        Initialize the dataset by loading and encoding identifier names.
        
        Args:
            readable_path (str): Path to file containing readable identifiers
            obfuscated_path (str): Path to file containing obfuscated identifiers
        """
        self.samples = []

        # Load readable identifiers (positive class, label=1)
        with open(readable_path) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.samples.append((encode(line), 1))

        # Load obfuscated identifiers (negative class, label=0)
        with open(obfuscated_path) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.samples.append((encode(line), 0))

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of (identifier, label) pairs
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (encoded_identifier, label) as PyTorch tensors
                - encoded_identifier: LongTensor of shape (MAX_LEN,)
                - label: FloatTensor of shape (1,) with value 0 or 1
        """
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.long),      # Encoded identifier sequence
            torch.tensor([y], dtype=torch.float32), # Binary label (0 or 1)
        )
