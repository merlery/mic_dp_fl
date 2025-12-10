# MIC-Based Transformation Method

## Overview

This implementation adds a **Maximum Information Coefficient (MIC)**-based transformation method as an alternative to the baseline **Input Normalization (IN)** method. Both methods are now available for comparison in the PrivateFL framework.

## What is MIC?

The Maximum Information Coefficient (MIC) is a measure of the strength of the relationship between two variables. It captures both linear and non-linear dependencies, making it useful for identifying which features are most informative for the target labels.

## Implementation Details

### Baseline Method (IN - Input Normalization)
- Uses learnable `gamma` (scale) and `beta` (shift) parameters
- Parameters are initialized randomly and learned during training
- Personalized per client through federated learning

### MIC-Based Method (MIC)
- Uses MIC to compute initial transformation weights
- Features with higher MIC scores (stronger relationship with labels) get higher weights
- Still learnable during training, but starts from MIC-informed initialization
- Personalized per client using MIC computed on client's local data

## New Components

### 1. `mic_utils.py`
- `compute_mic_matrix()`: Computes MIC scores between features and labels
- `compute_mic_weights()`: Generates transformation weights based on MIC scores
- `compute_mic_for_batch()`: Computes MIC weights for a batch of data

### 2. New Model Classes in `modelUtil.py`
- `MICNorm`: MIC-based input normalization layer
- `FeatureNorm_MIC`: MIC-based feature normalization layer
- Model variants with `_MIC` suffix:
  - `mnist_fully_connected_MIC`
  - `resnet18_MIC`
  - `alexnet_MIC`
  - `purchase_fully_connected_MIC`
  - `linear_model_DN_MIC`

## Usage

### Running Baseline Method (Original)
```bash
python FedAverage.py --data='mnist' --model='mnist_fully_connected_IN' --mode='CDP' --epsilon=2
```

### Running MIC-Based Method
```bash
python FedAverage.py --data='mnist' --model='mnist_fully_connected_MIC' --mode='CDP' --epsilon=2
```

### Comparing Both Methods
```bash
# Linux/Mac
cd script
bash compare_methods.sh --data=mnist --mode=CDP --epsilon=2

# Windows
cd script
compare_methods.bat
```

Or directly:
```bash
python compare_methods.py --data=mnist --mode=CDP --epsilon=2
```

## Installation

Make sure to install the `minepy` library:
```bash
pip install minepy
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Comparison Results

The comparison script (`compare_methods.py`) will:
1. Run the baseline (IN) method
2. Run the MIC-based method
3. Compare accuracy and performance
4. Save results to `log/comparison_*.csv`

## Model Selection Guide

| Dataset | Baseline Model | MIC Model |
|---------|---------------|-----------|
| MNIST | `mnist_fully_connected_IN` | `mnist_fully_connected_MIC` |
| CIFAR-10 | `resnet18_IN` | `resnet18_MIC` |
| CIFAR-100 | `resnet18_IN` | `resnet18_MIC` |
| Purchase | `purchase_fully_connected_IN` | `purchase_fully_connected_MIC` |

## Key Differences

| Aspect | Baseline (IN) | MIC-Based |
|--------|---------------|-----------|
| Initialization | Random | MIC-informed |
| Feature Selection | Implicit (learned) | Explicit (MIC-guided) |
| Non-linear Dependencies | Learned during training | Captured in initialization |
| Computational Cost | Lower | Higher (MIC computation) |

## Notes

- MIC computation can be computationally expensive for large datasets
- MIC parameters (`alpha=0.6`, `c=15`) are set to defaults but can be adjusted
- The MIC-based transformation is still learnable - MIC only provides initialization
- Both methods maintain the same privacy guarantees (differential privacy)

## References

- Original PrivateFL paper: Yang et al., USENIX Security 2023
- MIC: Reshef et al., "Detecting Novel Associations in Large Data Sets", Science 2011

