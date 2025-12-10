@echo off
REM PrivateFL Training Script for Windows (Batch File)
REM This script runs federated learning training from scratch

echo Starting PrivateFL training (FedAvg)...

REM Activate conda environment
call conda activate privatefl
if %ERRORLEVEL% NEQ 0 (
    echo Error: Could not activate 'privatefl' environment.
    echo Please run setup.bat first or manually activate: conda activate privatefl
    exit /b 1
)

REM Navigate to project root
cd ..

REM Default parameters - modify these as needed
set DATA=mnist
set NCLIENT=100
set NCLASS=10
set NCPC=2
set MODEL=mnist_fully_connected_IN
set MODE=CDP
set ROUND=60
set EPSILON=2
set SR=1
set LR=5e-3
set FLR=1e-2
set PHYSICAL_BS=64
set E=1

REM Run training
echo Running FedAverage.py with parameters:
echo   Data: %DATA%
echo   Clients: %NCLIENT%
echo   Classes: %NCLASS%
echo   Classes per client: %NCPC%
echo   Model: %MODEL%
echo   Mode: %MODE%
echo   Rounds: %ROUND%
echo   Epsilon: %EPSILON%
echo.

python FedAverage.py --data=%DATA% --nclient=%NCLIENT% --nclass=%NCLASS% --ncpc=%NCPC% --model=%MODEL% --mode=%MODE% --round=%ROUND% --epsilon=%EPSILON% --sr=%SR% --lr=%LR% --flr=%FLR% --physical_bs=%PHYSICAL_BS% --E=%E%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training completed successfully!
) else (
    echo.
    echo Training failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

pause

