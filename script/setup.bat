@echo off
REM PrivateFL Setup Script for Windows (Batch File)
REM This script sets up the conda environment and installs dependencies

echo Setting up PrivateFL environment...

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Create conda environment
echo Creating conda environment 'privatefl' with Python 3.8...
call conda create --name privatefl python=3.8 -y

REM Activate environment and install packages
echo Activating environment and installing packages...
call conda activate privatefl

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA 11.6...
echo Note: If you don't have a GPU, you may need to install CPU-only version instead.
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

REM Install requirements
echo Installing project requirements...
cd ..
pip install -r requirements.txt
cd script

echo.
echo Setup complete!
echo To activate the environment, run: conda activate privatefl
echo Then navigate to the project root and run the training scripts.

pause

