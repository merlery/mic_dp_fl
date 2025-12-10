@echo off
REM PrivateFL Transfer Learning Script for Windows (Batch File)
REM This script runs federated learning with frozen encoder (transfer learning)

echo Starting PrivateFL transfer learning (FedTransfer)...

REM Activate conda environment
call conda activate privatefl
if %ERRORLEVEL% NEQ 0 (
    echo Error: Could not activate 'privatefl' environment.
    echo Please run setup.bat first or manually activate: conda activate privatefl
    exit /b 1
)

REM Navigate to transfer directory
cd ..
cd transfer

REM Default parameters - modify these as needed
set DATA=cifar10
set NCLIENT=100
set NCLASS=10
set NCPC=2
set ENCODER=clip
set MODEL=linear_model_DN_IN
set MODE=LDP
set ROUND=20
set EPSILON=8
set SR=1
set LR=1e-1
set FLR=1e-1
set PHYSICAL_BS=64
set E=2
set BS=64

REM Run training
echo Running FedTransfer.py with parameters:
echo   Data: %DATA%
echo   Clients: %NCLIENT%
echo   Classes: %NCLASS%
echo   Classes per client: %NCPC%
echo   Encoder: %ENCODER%
echo   Model: %MODEL%
echo   Mode: %MODE%
echo   Rounds: %ROUND%
echo   Epsilon: %EPSILON%
echo   Physical batch size: %PHYSICAL_BS%
echo   Note: Reduce --physical_bs if facing CUDA out of memory
echo.

python FedTransfer.py --data=%DATA% --nclient=%NCLIENT% --nclass=%NCLASS% --ncpc=%NCPC% --encoder=%ENCODER% --model=%MODEL% --mode=%MODE% --round=%ROUND% --epsilon=%EPSILON% --sr=%SR% --lr=%LR% --flr=%FLR% --physical_bs=%PHYSICAL_BS% --E=%E% --bs=%BS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Transfer learning completed successfully!
) else (
    echo.
    echo Transfer learning failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

pause

