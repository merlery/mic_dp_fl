@echo off
REM Comparison script for baseline vs MIC methods (Windows)

echo Comparing Baseline (IN) vs MIC-based transformation methods

cd ..

python compare_methods.py --data=mnist --nclient=100 --nclass=10 --ncpc=2 --mode=CDP --round=60 --epsilon=2 --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=64 --E=1

pause

