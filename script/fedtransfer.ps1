# PrivateFL Transfer Learning Script for Windows
# This script runs federated learning with frozen encoder (transfer learning)

Write-Host "Starting PrivateFL transfer learning (FedTransfer)..." -ForegroundColor Green

# Check if conda environment is activated
$envName = $env:CONDA_DEFAULT_ENV
if ($envName -ne "privatefl") {
    Write-Host "Warning: 'privatefl' conda environment is not activated." -ForegroundColor Yellow
    Write-Host "Attempting to activate..." -ForegroundColor Yellow
    conda activate privatefl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Could not activate 'privatefl' environment." -ForegroundColor Red
        Write-Host "Please run: conda activate privatefl" -ForegroundColor Yellow
        exit 1
    }
}

# Navigate to transfer directory
$projectRoot = Split-Path -Parent $PSScriptRoot
$transferDir = Join-Path $projectRoot "transfer"
Set-Location $transferDir

# Default parameters (can be modified)
$data = "cifar10"
$nclient = 100
$nclass = 10
$ncpc = 2
$encoder = "clip"
$model = "linear_model_DN_IN"
$mode = "LDP"
$round = 20
$epsilon = 8
$sr = 1
$lr = 1e-1
$flr = 1e-1
$physical_bs = 64
$E = 2
$bs = 64

# Allow parameters to be passed as arguments
if ($args.Count -gt 0) {
    # Parse arguments if provided
    param(
        [string]$data = "cifar10",
        [int]$nclient = 100,
        [int]$nclass = 10,
        [int]$ncpc = 2,
        [string]$encoder = "clip",
        [string]$model = "linear_model_DN_IN",
        [string]$mode = "LDP",
        [int]$round = 20,
        [int]$epsilon = 8,
        [float]$sr = 1,
        [float]$lr = 1e-1,
        [float]$flr = 1e-1,
        [int]$physical_bs = 64,
        [int]$E = 2,
        [int]$bs = 64
    )
}

# Run training
Write-Host "Running FedTransfer.py with parameters:" -ForegroundColor Cyan
Write-Host "  Data: $data" -ForegroundColor White
Write-Host "  Clients: $nclient" -ForegroundColor White
Write-Host "  Classes: $nclass" -ForegroundColor White
Write-Host "  Classes per client: $ncpc" -ForegroundColor White
Write-Host "  Encoder: $encoder" -ForegroundColor White
Write-Host "  Model: $model" -ForegroundColor White
Write-Host "  Mode: $mode" -ForegroundColor White
Write-Host "  Rounds: $round" -ForegroundColor White
Write-Host "  Epsilon: $epsilon" -ForegroundColor White
Write-Host "  Physical batch size: $physical_bs" -ForegroundColor White
Write-Host "  Note: Reduce --physical_bs if facing CUDA out of memory" -ForegroundColor Yellow
Write-Host ""

python FedTransfer.py `
    --data=$data `
    --nclient=$nclient `
    --nclass=$nclass `
    --ncpc=$ncpc `
    --encoder=$encoder `
    --model=$model `
    --mode=$mode `
    --round=$round `
    --epsilon=$epsilon `
    --sr=$sr `
    --lr=$lr `
    --flr=$flr `
    --physical_bs=$physical_bs `
    --E=$E `
    --bs=$bs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTransfer learning completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nTransfer learning failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

