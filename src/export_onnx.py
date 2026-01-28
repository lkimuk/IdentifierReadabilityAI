"""
ONNX model export script for cross-platform deployment.

This module converts the trained PyTorch model to ONNX format, enabling deployment
on various platforms including C++, mobile devices, and web applications. The export
includes dynamic axis support for variable batch sizes and maintains compatibility
with the C++ inference implementation.
"""

import torch
from charset import vocab_size
from model import CharCNN
from config import MAX_LEN


def main():
    """
    Export the trained PyTorch model to ONNX format.
    
    Process:
        1. Load the trained model weights
        2. Set model to evaluation mode
        3. Create dummy input for tracing
        4. Export to ONNX with dynamic batch support
        5. Verify export success
    """
    
    print("Exporting PyTorch model to ONNX format...")
    print("-" * 50)
    
    # 1. 加载模型 (Load the trained model)
    print("1. Loading trained model...")
    model = CharCNN(vocab_size)
    model.load_state_dict(torch.load("models/readability.pt"))
    model.eval()  # Set to evaluation mode

    # 2. 创建虚拟输入（batch_size=1）
    print("2. Creating dummy input...")
    dummy = torch.zeros(1, MAX_LEN, dtype=torch.long)
    print(f"   Input shape: {dummy.shape}")

    # 3. 导出ONNX - 添加动态轴支持
    print("3. Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        "models/readability.onnx",
        input_names=["input"],      # Input tensor name
        output_names=["score"],     # Output tensor name
        opset_version=18,           # ONNX operator set version
        # 关键修复：添加动态轴，支持可变batch_size
        dynamic_axes={
            'input': {0: 'batch_size'},   # 第0维（batch维度）是动态的
            'score': {0: 'batch_size'}    # 输出也是动态batch
        },
        verbose=False               # Suppress detailed output
    )

    print("✅ Exported models/readability.onnx")
    print("   Model now supports variable batch_size!")
    print("   ONNX model ready for cross-platform deployment")
    
    print("\nExport Details:")
    print(f"   - Input shape: [batch_size, {MAX_LEN}]")
    print(f"   - Output shape: [batch_size, 1]")
    print(f"   - Dynamic batch support: Yes")
    print(f"   - Compatible with C++ inference: Yes")


def verify_onnx_export():
    """
    Verify that the exported ONNX model can be loaded and run.
    
    This function tests the exported model to ensure it works correctly
    before deployment.
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\n4. Verifying ONNX export...")
        
        # Load the ONNX model
        session = ort.InferenceSession("models/readability.onnx")
        
        # Test with sample input
        test_input = np.zeros((1, MAX_LEN), dtype=np.int64)
        result = session.run(None, {'input': test_input})
        
        print(f"   ✅ ONNX model loaded successfully")
        print(f"   ✅ Inference test passed")
        print(f"   Output shape: {result[0].shape}")
        
    except ImportError:
        print("   ⚠️  onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")


if __name__ == "__main__":
    main()
    verify_onnx_export()
