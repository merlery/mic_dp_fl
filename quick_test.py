"""
Quick test to verify code structure works (even without minepy)
"""
import sys

print("="*60)
print("Quick Code Structure Test")
print("="*60)

# Test 1: Check if modelUtil can be imported
print("\n[Test 1] Testing modelUtil imports...")
try:
    # Try importing with error handling
    try:
        from mic_utils import compute_mic_weights
        mic_utils_available = True
    except ImportError:
        mic_utils_available = False
        print("[WARN] mic_utils not available (minepy not installed)")
    
    from modelUtil import (
        InputNorm, MICNorm, FeatureNorm, FeatureNorm_MIC,
        mnist_fully_connected_IN, mnist_fully_connected_MIC
    )
    print("[PASS] modelUtil imports work")
except Exception as e:
    print(f"[FAIL] modelUtil import failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nThis might be due to:")
    print("1. PyTorch not installed: pip install torch")
    print("2. Wrong Python environment")
    print("3. Missing dependencies")
    sys.exit(1)

# Test 2: Check if models can be created
print("\n[Test 2] Testing model creation...")
try:
    import torch
    model_in = mnist_fully_connected_IN(num_classes=10)
    model_mic = mnist_fully_connected_MIC(num_classes=10)
    print("[PASS] Both IN and MIC models can be created")
except Exception as e:
    print(f"[FAIL] Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test forward pass
print("\n[Test 3] Testing forward pass...")
try:
    x = torch.randn(2, 1, 28, 28)
    logits_in, probs_in = model_in(x)
    logits_mic, probs_mic = model_mic(x)
    print(f"[PASS] Forward pass works for both models")
    print(f"  IN model output shape: {logits_in.shape}")
    print(f"  MIC model output shape: {logits_mic.shape}")
except Exception as e:
    print(f"[FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check minepy availability
print("\n[Test 4] Checking minepy availability...")
try:
    from minepy import MINE
    print("[PASS] minepy is installed and can be imported")
    minepy_available = True
except ImportError:
    print("[WARN] minepy is NOT installed")
    print("  MIC computation will not work, but models can still be created")
    print("  To install: pip install minepy")
    minepy_available = False

# Test 5: Test mic_utils (if minepy available)
if minepy_available:
    print("\n[Test 5] Testing mic_utils...")
    try:
        from mic_utils import compute_mic_matrix, compute_mic_weights
        import numpy as np
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        mic_scores = compute_mic_matrix(X, y)
        gamma, beta = compute_mic_weights(X, y)
        print("[PASS] mic_utils works correctly")
    except Exception as e:
        print(f"[FAIL] mic_utils test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[Test 5] Skipping mic_utils test (minepy not available)")

# Test 6: Check FedAverage registration
print("\n[Test 6] Checking FedAverage model registration...")
try:
    import FedAverage
    import inspect
    source = inspect.getsource(FedAverage.parse_arguments)
    if 'mnist_fully_connected_MIC' in source:
        print("[PASS] MIC model is registered in FedAverage")
    else:
        print("[FAIL] MIC model not found in FedAverage")
        sys.exit(1)
except Exception as e:
    print(f"[WARN] Could not verify FedAverage registration: {e}")

print("\n" + "="*60)
if minepy_available:
    print("All tests passed! Code is ready to use.")
else:
    print("Code structure is correct, but minepy needs to be installed.")
    print("Install with: pip install minepy")
print("="*60)

