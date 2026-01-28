"""
ONNX model validation and testing script.

This module provides comprehensive testing for the exported ONNX model to ensure
it produces identical results to the original PyTorch model. It includes both
single inference and batch inference tests, with detailed comparison and validation.
"""

import torch
import onnxruntime as ort
import numpy as np
from charset import encode


def test_onnx_model():
    """
    Test the exported ONNX model against the original PyTorch model.
    
    This function:
        1. Loads and tests the PyTorch model
        2. Loads and tests the ONNX model
        3. Compares results to ensure numerical equivalence
        4. Validates that the models produce identical predictions
    
    Returns:
        bool: True if models match within tolerance, False otherwise
    """
    print("Testing exported ONNX model...")
    
    # 1. 先测试PyTorch模型 (Test PyTorch model first)
    print("\n1. Testing PyTorch model:")
    from model import CharCNN
    from charset import vocab_size
    
    pt_model = CharCNN(vocab_size)
    pt_model.load_state_dict(torch.load("models/readability.pt"))
    pt_model.eval()
    
    # Test identifiers with varying readability levels
    test_names = ["strlen", "UserManager", "aZ9f", "IlIlI"]
    pt_results = {}
    
    with torch.no_grad():
        for name in test_names:
            encoded = encode(name)
            input_tensor = torch.tensor([encoded], dtype=torch.long)
            score = pt_model(input_tensor).item()
            pt_results[name] = score
            print(f"   {name:15} -> {score:.4f}")
    
    # 2. 测试ONNX模型 (Test ONNX model)
    print("\n2. Testing ONNX model:")
    ort_session = ort.InferenceSession("models/readability.onnx")
    
    onnx_results = {}
    all_match = True
    
    for name in test_names:
        # Encode identifier for ONNX input
        encoded = encode(name)
        input_tensor = np.array([encoded], dtype=np.int64)
        
        # Run ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_score = ort_outputs[0][0][0]
        
        # Compare with PyTorch result
        pt_score = pt_results[name]
        diff = abs(onnx_score - pt_score)
        
        # Check if results match within tolerance
        status = "✅" if diff < 0.001 else "⚠️"
        if diff >= 0.001:
            all_match = False
            
        print(f"   {name:15} -> {onnx_score:.4f} (PyTorch: {pt_score:.4f}, diff: {diff:.6f}) {status}")
        onnx_results[name] = onnx_score
    
    if all_match:
        print("\n✅ ONNX model works correctly! All results match PyTorch within tolerance.")
        return True
    else:
        print("\n❌ ONNX model validation failed! Some results differ significantly.")
        return False


def test_batch_inference():
    """
    Test batch inference functionality of the ONNX model.
    
    This function validates that the ONNX model can process multiple
    identifiers in a single batch, which is important for performance
    in production environments.
    """
    print("\n3. Testing batch inference:")
    
    ort_session = ort.InferenceSession("models/readability.onnx")
    
    # 创建批量输入 (Create batch input)
    batch_names = ["strlen", "atoi", "memcpy", "aZ9f", "IlIlI"]
    batch_input = []
    
    for name in batch_names:
        encoded = encode(name)
        batch_input.append(encoded)
    
    # 转换为numpy数组 (Convert to numpy array)
    batch_array = np.array(batch_input, dtype=np.int64)
    print(f"   Batch shape: {batch_array.shape}")  # 应该是 (5, 20)
    
    # 运行推理 (Run inference)
    ort_inputs = {ort_session.get_inputs()[0].name: batch_array}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("   Results:")
    for i, name in enumerate(batch_names):
        score = ort_outputs[0][i][0]
        readable = "Readable" if score > 0.5 else "Obfuscated"
        print(f"   {name:15} -> {score:.4f} ({readable})")


def test_edge_cases():
    """
    Test the model with edge cases and special inputs.
    
    This function tests various edge cases to ensure robustness:
        - Empty strings
        - Very short identifiers
        - Very long identifiers (truncated)
        - Identifiers with unknown characters
    """
    print("\n4. Testing edge cases:")
    
    ort_session = ort.InferenceSession("models/readability.onnx")
    
    edge_cases = [
        "",              # Empty string
        "a",             # Single character
        "very_long_identifier_name_that_exceeds_max_length",  # Too long
        "test@#$%",      # Contains invalid characters
        "123",           # Only numbers
        "___",           # Only underscores
    ]
    
    for case in edge_cases:
        try:
            encoded = encode(case)
            input_tensor = np.array([encoded], dtype=np.int64)
            
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
            ort_outputs = ort_session.run(None, ort_inputs)
            score = ort_outputs[0][0][0]
            
            print(f"   '{case:30}' -> {score:.4f}")
        except Exception as e:
            print(f"   '{case:30}' -> ERROR: {e}")


def benchmark_performance():
    """
    Benchmark the performance of the ONNX model.
    
    This function measures inference speed to provide performance metrics
    for production deployment planning.
    """
    print("\n5. Performance benchmark:")
    
    import time
    
    ort_session = ort.InferenceSession("models/readability.onnx")
    
    # Create test batch
    test_batch = []
    for i in range(100):
        test_batch.append(encode(f"test_identifier_{i}"))
    
    batch_array = np.array(test_batch, dtype=np.int64)
    
    # Warm up
    ort_inputs = {ort_session.get_inputs()[0].name: batch_array}
    ort_session.run(None, ort_inputs)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    throughput = len(test_batch) / avg_time
    
    print(f"   Batch size: {len(test_batch)}")
    print(f"   Average time: {avg_time:.4f} seconds")
    print(f"   Throughput: {throughput:.2f} identifiers/second")


if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Model Validation Suite")
    print("=" * 60)
    
    # Run all tests
    success = test_onnx_model()
    test_batch_inference()
    test_edge_cases()
    benchmark_performance()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! ONNX model is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the model export.")
    print("=" * 60)
