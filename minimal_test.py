"""
Minimal test to isolate issues
"""
import sys

print("Testing basic imports...")

# Test 1: Basic Python
print("1. Python version:", sys.version)

# Test 2: PyTorch
try:
    import torch
    print("2. PyTorch:", torch.__version__)
except Exception as e:
    print(f"2. PyTorch FAILED: {e}")
    sys.exit(1)

# Test 3: torch.nn.functional
try:
    from torch.nn.functional import relu
    print("3. torch.nn.functional: OK")
except Exception as e:
    print(f"3. torch.nn.functional FAILED: {e}")
    sys.exit(1)

# Test 4: modelUtil (with graceful mic_utils handling)
print("4. Testing modelUtil import...")
try:
    # First check if mic_utils can be imported
    try:
        from mic_utils import compute_mic_weights
        print("   mic_utils: Available")
    except ImportError as e:
        print(f"   mic_utils: Not available ({e}) - this is OK if minepy not installed")
    
    # Now try modelUtil
    from modelUtil import InputNorm, MICNorm
    print("4. modelUtil: OK")
except Exception as e:
    print(f"4. modelUtil FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create a model
print("5. Testing model creation...")
try:
    from modelUtil import mnist_fully_connected_MIC
    model = mnist_fully_connected_MIC(num_classes=10)
    print("5. Model creation: OK")
except Exception as e:
    print(f"5. Model creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All basic tests passed!")
print("="*50)
print("\nNote: If minepy is not installed, MIC computation won't work,")
print("but models can still be created and used with default initialization.")

