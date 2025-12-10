# PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

This repository contains the PyTorch implementation of **PrivateFL**, which introduces a personalized data transformation approach for differentially private federated learning. The implementation includes both the original **Input Normalization (IN)** method and a new **MIC-based transformation** method for comparison.

## Features

- ✅ **Baseline Method (IN)**: Original Input Normalization approach
- ✅ **MIC-Based Method**: Alternative using Maximum Information Coefficient (uses scikit-learn mutual information)
- ✅ **Differential Privacy**: Supports both CDP and LDP modes
- ✅ **Multiple Datasets**: MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST, EMNIST, Purchase, CHMNIST
- ✅ **Transfer Learning**: Support for frozen encoder training with ResNeXt, SimCLR, CLIP

## Quick Start

### Prerequisites

- Python 3.8
- Conda (Miniconda or Anaconda)
- CUDA-capable GPU (recommended)

### Installation

#### Option 1: Using Conda (Recommended)

```bash
cd script
bash setup.sh
```

#### Option 2: Manual Setup

```bash
conda create --name privatefl python=3.8
conda activate privatefl
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

**Note**: `uvloop` in requirements.txt may fail on Windows - it's optional and can be skipped.

### Running Experiments

#### Baseline Method (IN)

```bash
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 \
    --model='mnist_fully_connected_IN' --mode='CDP' --round=60 --epsilon=2 \
    --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=64 --E=1
```

#### MIC-Based Method

```bash
python FedAverage.py --data='mnist' --nclient=100 --nclass=10 --ncpc=2 \
    --model='mnist_fully_connected_MIC' --mode='CDP' --round=60 --epsilon=2 \
    --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=64 --E=1
```

#### Compare Both Methods

```bash
python compare_methods.py --data=mnist --mode=CDP --epsilon=2
```

## Project Structure

```
.
├── FedAverage.py          # Main training script
├── FedUser.py             # Client-side implementation
├── FedServer.py           # Server-side implementation
├── modelUtil.py           # Model definitions (IN and MIC)
├── mic_utils.py           # MIC computation utilities
├── datasets.py            # Dataset loaders
├── dataloader.py          # Data loading utilities
├── compare_methods.py     # Comparison script
├── script/                # Setup and run scripts
│   ├── setup.sh          # Setup script
│   ├── fedavg.sh         # Run baseline
│   └── fedtransfer.sh    # Run transfer learning
├── transfer/              # Transfer learning code
│   ├── FedTransfer.py    # Transfer learning script
│   └── ExtractFeature.py # Feature extraction
└── data/                  # Dataset directory
```

## Available Models

### Baseline (IN) Models
- `mnist_fully_connected_IN`
- `resnet18_IN`
- `alexnet_IN`
- `purchase_fully_connected_IN`
- `linear_model_DN_IN` (for transfer learning)

### MIC-Based Models
- `mnist_fully_connected_MIC`
- `resnet18_MIC`
- `alexnet_MIC`
- `purchase_fully_connected_MIC`
- `linear_model_DN_MIC` (for transfer learning)

## Parameters

- `--data`: Dataset (`mnist`, `cifar10`, `cifar100`, `fashionmnist`, `emnist`, `purchase`, `chmnist`)
- `--nclient`: Number of clients (default: 100)
- `--nclass`: Number of classes (default: 10)
- `--ncpc`: Number of classes per client (default: 2)
- `--model`: Model architecture (see choices above)
- `--mode`: Privacy mode (`CDP` or `LDP`)
- `--round`: Number of training rounds (default: 150)
- `--epsilon`: Privacy budget (default: 8)
- `--lr`: Learning rate (default: 0.1)
- `--flr`: Federated learning rate (default: 0.1)
- `--physical_bs`: Physical batch size for Opacus (reduce if CUDA OOM)

## MIC Implementation

The MIC-based transformation uses **scikit-learn mutual information** as an alternative to minepy (which requires C++ compilation). This provides similar functionality without compilation requirements.

See [MIC_METHOD_README.md](MIC_METHOD_README.md) for detailed information.

## Citation

```bibtex
@inproceedings{yangprivatefl,
  title={PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation},
  author={Yang, Yuchen and Hui, Bo and Yuan, Haolin and Gong, Neil and Cao, Yinzhi},
  booktitle = {Proceedings of the USENIX Security Symposium (Usenix'23)},
  year = {2023}
}
```

## License

See LICENSE file for details.

## Troubleshooting

- **Opacus version issue**: If you see `No module named 'torch.func'`, install compatible opacus: `pip install opacus==1.0.0`
- **Windows PowerShell**: Use `.\script.ps1` or `powershell -ExecutionPolicy Bypass -File script.ps1`
- **CUDA OOM**: Reduce `--physical_bs` parameter

## For ORNL Frontier Users

See [FRONTIER_SETUP.md](FRONTIER_SETUP.md) for Frontier-specific instructions.

