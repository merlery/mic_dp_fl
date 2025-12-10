#!/bin/bash
# Comparison script for baseline vs MIC methods

echo "Comparing Baseline (IN) vs MIC-based transformation methods"

# Default parameters
DATA="mnist"
NCLIENT=100
NCLASS=10
NCPC=2
MODE="CDP"
ROUND=60
EPSILON=2
SR=1
LR=5e-3
FLR=1e-2
PHYSICAL_BS=64
E=1

# Parse arguments if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA="$2"
            shift 2
            ;;
        --nclient)
            NCLIENT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

cd ..
python compare_methods.py \
    --data=$DATA \
    --nclient=$NCLIENT \
    --nclass=$NCLASS \
    --ncpc=$NCPC \
    --mode=$MODE \
    --round=$ROUND \
    --epsilon=$EPSILON \
    --sr=$SR \
    --lr=$LR \
    --flr=$FLR \
    --physical_bs=$PHYSICAL_BS \
    --E=$E

