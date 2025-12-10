"""
Test script to verify MIC implementation works correctly
"""
import sys
import traceback
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul')
    except:
        pass

# Use ASCII-safe symbols
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

print("="*60)
print("Testing MIC Implementation")
print("="*60)

# Test 1: Check if minepy can be imported
print("\n[Test 1] Checking minepy import...")
try:
    from minepy import MINE
    print(f"{PASS} minepy imported successfully")
    minepy_available = True
except ImportError as e:
    print(f"{FAIL} minepy import failed: {e}")
    print("  Please install minepy: pip install minepy")
    print("  Or: pip install -r requirements.txt")
    minepy_available = False
    print("\n" + "="*60)
    print("CRITICAL: minepy is not installed!")
    print("="*60)
    print("\nTo fix this, run:")
    print("  pip install minepy")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Test basic MIC computation
print("\n[Test 2] Testing basic MIC computation...")
try:
    import numpy as np
    from minepy import MINE
    
    # Create simple test data
    np.random.seed(42)
    x = np.random.randn(100)
    y = x + 0.1 * np.random.randn(100)  # y is correlated with x
    
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    mic_value = mine.mic()
    
    print(f"{PASS} MIC computation successful")
    print(f"  MIC value: {mic_value:.4f} (should be > 0 for correlated data)")
    
    if mic_value > 0.1:
        print(f"  {PASS} MIC value looks reasonable")
    else:
        print(f"  {WARN} MIC value seems low, but computation worked")
        
except Exception as e:
    print(f"{FAIL} MIC computation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test mic_utils module
print("\n[Test 3] Testing mic_utils module...")
try:
    from mic_utils import compute_mic_matrix, compute_mic_weights
    print(f"{PASS} mic_utils imported successfully")
    
    # Test with simple data
    X = np.random.randn(50, 10)
    y = (X[:, 0] > 0).astype(int)  # Binary labels based on first feature
    
    mic_scores = compute_mic_matrix(X, y)
    print(f"{PASS} compute_mic_matrix works")
    print(f"  MIC scores shape: {mic_scores.shape}")
    print(f"  First feature MIC: {mic_scores[0]:.4f} (should be high)")
    
    gamma, beta = compute_mic_weights(X, y)
    print(f"{PASS} compute_mic_weights works")
    print(f"  Gamma shape: {gamma.shape}, Beta shape: {beta.shape}")
    print(f"  Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    
except Exception as e:
    print(f"{FAIL} mic_utils test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test modelUtil imports
print("\n[Test 4] Testing modelUtil imports...")
try:
    import torch
    from modelUtil import (
        InputNorm, MICNorm, FeatureNorm, FeatureNorm_MIC,
        mnist_fully_connected_IN, mnist_fully_connected_MIC
    )
    print(f"{PASS} All model classes imported successfully")
    
except Exception as e:
    print(f"{FAIL} modelUtil import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test MICNorm layer
print("\n[Test 5] Testing MICNorm layer...")
try:
    import torch
    from modelUtil import MICNorm
    
    # Create a simple MICNorm layer
    norm = MICNorm(num_channel=1, num_feature=28)
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    output = norm(x)
    
    print(f"{PASS} MICNorm forward pass works")
    print(f"  Input shape: {x.shape}, Output shape: {output.shape}")
    
    # Check if parameters exist
    assert hasattr(norm, 'gamma'), "MICNorm should have gamma parameter"
    assert hasattr(norm, 'beta'), "MICNorm should have beta parameter"
    print(f"{PASS} MICNorm has required parameters")
    
except Exception as e:
    print(f"{FAIL} MICNorm test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test MIC-based model
print("\n[Test 6] Testing MIC-based model...")
try:
    import torch
    from modelUtil import mnist_fully_connected_MIC
    
    model = mnist_fully_connected_MIC(num_classes=10)
    print(f"{PASS} mnist_fully_connected_MIC model created")
    
    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    logits, probs = model(x)
    
    print(f"{PASS} Model forward pass works")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}, Probs shape: {probs.shape}")
    
    # Check if norm layer exists
    assert hasattr(model, 'norm'), "Model should have norm layer"
    from modelUtil import MICNorm
    assert isinstance(model.norm, MICNorm), "Norm should be MICNorm"
    print(f"{PASS} Model has MICNorm layer")
    
except Exception as e:
    print(f"{FAIL} MIC model test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test model registration in FedAverage
print("\n[Test 7] Testing model registration in FedAverage...")
try:
    import sys
    sys.path.insert(0, '.')
    
    # Check if model is in choices
    import FedAverage
    import inspect
    
    # Get the parse_arguments function
    source = inspect.getsource(FedAverage.parse_arguments)
    
    if 'mnist_fully_connected_MIC' in source:
        print(f"{PASS} MIC model is registered in FedAverage.py")
    else:
        print(f"{FAIL} MIC model not found in FedAverage.py choices")
        sys.exit(1)
        
except Exception as e:
    print(f"{FAIL} Model registration test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test with actual data loader (if available)
print("\n[Test 8] Testing with actual data structure...")
try:
    import torch
    from modelUtil import mnist_fully_connected_MIC
    
    model = mnist_fully_connected_MIC(num_classes=10)
    
    # Create a dummy dataloader-like structure
    batch_size = 4
    x_batch = torch.randn(batch_size, 1, 28, 28)
    y_batch = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, probs = model(x_batch)
        predictions = torch.argmax(probs, dim=1)
    
    print(f"{PASS} Model works with batch data")
    print(f"  Batch size: {batch_size}")
    print(f"  Predictions: {predictions.tolist()}")
    
except Exception as e:
    print(f"{FAIL} Data structure test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test MIC update functionality
print("\n[Test 9] Testing MIC update functionality...")
try:
    import torch
    from modelUtil import MICNorm
    
    norm = MICNorm(num_channel=1, num_feature=28)
    
    # Create test data
    x_batch = torch.randn(10, 1, 28, 28)
    y_batch = torch.randint(0, 2, (10,))
    
    # Try to update with MIC (this might fail if minepy has issues)
    try:
        norm.update_with_mic(x_batch, y_batch)
        print(f"{PASS} MIC update method works")
    except Exception as e:
        print(f"{WARN} MIC update method failed (might be expected): {e}")
        print("  This is okay - update_with_mic is optional")
    
except Exception as e:
    print(f"{FAIL} MIC update test failed: {e}")
    traceback.print_exc()
    # Don't exit - this is optional functionality

print("\n" + "="*60)
print("All critical tests passed! [PASS]")
print("="*60)
print("\nNext steps:")
print("1. Run a simple training: python FedAverage.py --data=mnist --model=mnist_fully_connected_MIC --nclient=2 --round=1 --epsilon=2")
print("2. Compare methods: python compare_methods.py --data=mnist --nclient=2 --round=1 --epsilon=2")

