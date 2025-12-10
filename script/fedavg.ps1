# PrivateFL Training Script for Windows
# This script runs federated learning training from scratch

Write-Host "Starting PrivateFL training (FedAvg)..." -ForegroundColor Green

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

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Default parameters (can be modified)
$data = "mnist"
$nclient = 100
$nclass = 10
$ncpc = 2
$model = "mnist_fully_connected_IN"
$mode = "CDP"
$round = 60
$epsilon = 2
$sr = 1
$lr = 5e-3
$flr = 1e-2
$physical_bs = 64
$E = 1

# Allow parameters to be passed as arguments
if ($args.Count -gt 0) {
    # Parse arguments if provided
    # Example: .\fedavg.ps1 -data "cifar10" -model "resnet18_IN"
    param(
        [string]$data = "mnist",
        [int]$nclient = 100,
        [int]$nclass = 10,
        [int]$ncpc = 2,
        [string]$model = "mnist_fully_connected_IN",
        [string]$mode = "CDP",
        [int]$round = 60,
        [int]$epsilon = 2,
        [float]$sr = 1,
        [float]$lr = 5e-3,
        [float]$flr = 1e-2,
        [int]$physical_bs = 64,
        [int]$E = 1
    )
}

# Run training
Write-Host "Running FedAverage.py with parameters:" -ForegroundColor Cyan
Write-Host "  Data: $data" -ForegroundColor White
Write-Host "  Clients: $nclient" -ForegroundColor White
Write-Host "  Classes: $nclass" -ForegroundColor White
Write-Host "  Classes per client: $ncpc" -ForegroundColor White
Write-Host "  Model: $model" -ForegroundColor White
Write-Host "  Mode: $mode" -ForegroundColor White
Write-Host "  Rounds: $round" -ForegroundColor White
Write-Host "  Epsilon: $epsilon" -ForegroundColor White
Write-Host ""

python FedAverage.py `
    --data=$data `
    --nclient=$nclient `
    --nclass=$nclass `
    --ncpc=$ncpc `
    --model=$model `
    --mode=$mode `
    --round=$round `
    --epsilon=$epsilon `
    --sr=$sr `
    --lr=$lr `
    --flr=$flr `
    --physical_bs=$physical_bs `
    --E=$E

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTraining completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nTraining failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

