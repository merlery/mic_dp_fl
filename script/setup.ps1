# PrivateFL Setup Script for Windows
# This script sets up the conda environment and installs dependencies

Write-Host "Setting up PrivateFL environment..." -ForegroundColor Green

# Check if conda is available
$condaAvailable = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaAvailable) {
    Write-Host "Error: Conda is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Miniconda or Anaconda first." -ForegroundColor Yellow
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Create conda environment
Write-Host "Creating conda environment 'privatefl' with Python 3.8..." -ForegroundColor Cyan
conda create --name privatefl python=3.8 -y

# Activate environment and install packages
Write-Host "Activating environment and installing packages..." -ForegroundColor Cyan
conda activate privatefl

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA 11.6..." -ForegroundColor Cyan
Write-Host "Note: If you don't have a GPU, you may need to install CPU-only version instead." -ForegroundColor Yellow
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
Write-Host "Installing project requirements..." -ForegroundColor Cyan
$requirementsPath = Join-Path $PSScriptRoot "..\requirements.txt"
pip install -r $requirementsPath

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: conda activate privatefl" -ForegroundColor Yellow
Write-Host "Then navigate to the project root and run the training scripts." -ForegroundColor Yellow

