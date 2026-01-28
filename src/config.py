# Configuration file for the obfuscation readability detection model
# This file contains all hyperparameters and constants used throughout the project

# Maximum length for identifier names (padded/truncated to this length)
MAX_LEN = 20

# Character vocabulary for encoding identifiers
# Includes lowercase letters, uppercase letters, digits, and underscore
# This defines the valid character set for the model
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"  # Lowercase letters
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Uppercase letters  
    "0123456789_"                 # Digits and underscore
)

# Neural network architecture parameters
EMBED_DIM = 16      # Embedding dimension for character representations
NUM_FILTERS = 64    # Number of convolutional filters in the CNN
KERNEL_SIZE = 3     # Size of convolutional kernel

# Training parameters
BATCH_SIZE = 128    # Number of samples per training batch
EPOCHS = 500        # Number of training epochs
LR = 1e-3           # Learning rate for Adam optimizer
