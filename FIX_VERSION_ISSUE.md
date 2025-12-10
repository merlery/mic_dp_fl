# Fixing Version Compatibility Issue

## Problem
The error `No module named 'torch.func'` occurs because:
- Your PyTorch version: 1.13.1
- Opacus version in requirements: 1.1.2 (or newer)
- Newer opacus requires PyTorch 2.0+ (which has `torch.func`)

## Solutions

### Option 1: Use Compatible Opacus Version (Recommended)
Downgrade opacus to a version compatible with PyTorch 1.13.1:

```bash
pip install opacus==1.1.2
```

Actually, let's check which opacus version works with PyTorch 1.13.1. Try:

```bash
pip install opacus==1.0.0
```

Or check opacus documentation for PyTorch 1.13 compatibility.

### Option 2: Upgrade PyTorch (If Possible)
Upgrade to PyTorch 2.0+:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
```

**Warning**: This might break other parts of the code that expect PyTorch 1.13.1.

### Option 3: Test Without Opacus (For MIC Testing Only)
For testing MIC implementation without differential privacy:

1. Comment out opacus imports in modelUtil.py
2. Comment out opacus-related code in FedUser.py and FedServer.py
3. Run tests without privacy features

## Quick Fix Command

Try this first:
```bash
pip uninstall opacus -y
pip install opacus==1.0.0
```

Then test again:
```bash
python minimal_test.py
```

## Verification

After fixing, verify with:
```bash
python -c "import torch; import opacus; print('PyTorch:', torch.__version__, 'Opacus:', opacus.__version__)"
```

