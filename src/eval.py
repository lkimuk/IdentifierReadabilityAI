"""
Evaluation script for testing the trained obfuscation readability detection model.

This module provides functionality to load a trained model and evaluate it on
new identifier names. It demonstrates how to use the model for inference and
displays the readability scores for sample identifiers.
"""

import torch
from charset import encode, vocab_size
from model import CharCNN


def test(names):
    """
    Test the trained model on a list of identifier names.
    
    Args:
        names (list): List of identifier names to evaluate
        
    Process:
        1. Load the trained model weights
        2. Set model to evaluation mode
        3. Encode each identifier name
        4. Run inference to get readability scores
        5. Display results with interpretation
    """
    
    # Initialize model with same architecture as training
    model = CharCNN(vocab_size)
    
    # Load trained weights from file
    model.load_state_dict(torch.load("models/readability.pt"))
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    print("Identifier Readability Evaluation")
    print("=" * 40)
    print(f"{'Identifier':<20} {'Score':<8} {'Classification'}")
    print("-" * 40)
    
    # Evaluate each identifier
    with torch.no_grad():  # Disable gradient computation for inference
        for name in names:
            # Encode identifier to numerical sequence
            encoded = encode(name)
            
            # Convert to tensor and add batch dimension
            x = torch.tensor([encoded], dtype=torch.long)
            
            # Run inference
            score = model(x).item()
            
            # Classify based on threshold (0.5)
            classification = "Readable" if score > 0.5 else "Obfuscated"
            
            # Display results
            print(f"{name:<20} {score:<8.3f} {classification}")


def evaluate_identifier(identifier_name):
    """
    Evaluate a single identifier and return its readability score.
    
    Args:
        identifier_name (str): The identifier name to evaluate
        
    Returns:
        float: Readability score between 0 and 1
        
    Example:
        >>> score = evaluate_identifier("strlen")
        >>> print(f"Readability score: {score:.3f}")
    """
    model = CharCNN(vocab_size)
    model.load_state_dict(torch.load("models/readability.pt"))
    model.eval()

    with torch.no_grad():
        encoded = encode(identifier_name)
        x = torch.tensor([encoded], dtype=torch.long)
        score = model(x).item()
    
    return score


if __name__ == "__main__":
    # Test with sample identifiers
    test_identifiers = [
        "strlen",      # Standard library function (should be readable)
        "atoi",        # Standard library function (should be readable)
        "ctxl",        # Abbreviated context (borderline)
        "UserManager", # CamelCase class name (should be readable)
        "aZ9f",        # Random characters (should be obfuscated)
        "IlIlI",       # Confusing characters (should be obfuscated)
        "xQp2",        # Random characters (should be obfuscated)
    ]
    
    test(test_identifiers)
    
    print("\n" + "=" * 40)
    print("Score Interpretation:")
    print("  0.0 - 0.5: Obfuscated (hard to read)")
    print("  0.5 - 1.0: Readable (easy to understand)")
    print("  Higher scores indicate better readability")
