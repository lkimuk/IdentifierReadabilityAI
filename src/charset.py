"""
Character encoding utilities for the obfuscation readability detection model.

This module provides functionality to convert identifier names (strings) into 
numerical sequences that can be processed by the neural network model.
The encoding is consistent with the C++ implementation for cross-platform compatibility.
"""

from config import CHARS, MAX_LEN

# Padding token ID (0) - used for sequences shorter than MAX_LEN
PAD = 0

# Character-to-index mapping dictionary
# Characters are mapped to indices 1-63 (0 is reserved for padding)
# This matches the C++ implementation exactly
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}

# Vocabulary size includes all characters plus the padding token
vocab_size = len(char2idx) + 1


def encode(name: str):
    """
    Encode an identifier name into a numerical sequence.
    
    Args:
        name (str): The identifier name to encode
        
    Returns:
        list: A list of integers representing the encoded identifier
        
    Process:
        1. Convert each character to its corresponding index
        2. Unknown characters are mapped to PAD (0)
        3. Truncate sequences longer than MAX_LEN
        4. Pad sequences shorter than MAX_LEN with PAD tokens
        
    Example:
        encode("strlen") -> [38, 45, 49, 44, 44, 37, 46, 0, 0, ...] (length MAX_LEN)
    """
    # Convert characters to indices, unknown chars become PAD (0)
    seq = [char2idx.get(c, 0) for c in name[:MAX_LEN]]
    
    # Pad with zeros if sequence is shorter than MAX_LEN
    if len(seq) < MAX_LEN:
        seq += [PAD] * (MAX_LEN - len(seq))
    
    return seq

if __name__ == "__main__":
    print(f"Vocabulary Size: {vocab_size}")
    
    # 验证编码结果
    encoded_name = encode("ExampleName")
    print(f"Encoded 'ExampleName': ", end="")
    
    # 打印所有编码值
    for idx in encoded_name:
        print(f"{idx} ", end="")
    print()  # 换行
    
    # 输出详细信息
    print(f"\n详细验证信息:")
    print(f"CHARS: {CHARS}")
    print(f"CHARS长度: {len(CHARS)}")
    print(f"MAX_LEN: {MAX_LEN}")
    
    # 验证"ExampleName"中每个字符的映射
    print(f"\n'ExampleName'中字符的映射:")
    for i, c in enumerate("ExampleName"):
        idx = char2idx.get(c, PAD)
        print(f"  '{c}' -> {idx}")
    
    # 检查前10个编码值
    print(f"\n前10个编码值: {encoded_name[:10]}")
    print(f"总序列长度: {len(encoded_name)}")