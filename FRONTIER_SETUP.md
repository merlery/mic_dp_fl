# ORNL Frontier Setup Guide

This guide provides step-by-step instructions for setting up and running PrivateFL on ORNL Frontier supercomputer.

## Prerequisites

- ORNL Frontier account
- Access to Frontier login nodes
- Basic familiarity with SLURM

## Step 1: Clone Repository on Frontier

```bash
# SSH to Frontier
ssh username@frontier.olcf.ornl.gov

# Navigate to your work directory
cd $PROJWORK/<your_project_id>

# Clone the repository
git clone <your_github_repo_url>
cd usenix2026  # or your repo name
```

## Step 2: Set Up Environment

### Option A: Using Conda (Recommended)

```bash
# Load modules
module load PrgEnv-gnu
module load conda

# Create conda environment
conda create --name privatefl python=3.8 -y
conda activate privatefl

# Install PyTorch with CUDA (Frontier uses CUDA 11.x)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
pip install -r requirements.txt
```

### Option B: Using Frontier's Python Environment

```bash
# Load modules
module load PrgEnv-gnu
module load python/3.8-anaconda3

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Step 3: Prepare Data

```bash
# If you need to download datasets, do it on login node or in a job
# For MNIST/CIFAR, they will download automatically
# For custom datasets, place them in the data/ directory
```

## Step 4: Create SLURM Job Script

Create a file `run_experiment.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=privatefl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=gpu

# Load modules
module load PrgEnv-gnu
module load conda  # or python/3.8-anaconda3

# Activate environment
conda activate privatefl  # or: source venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run experiment
python FedAverage.py \
    --data='mnist' \
    --nclient=100 \
    --nclass=10 \
    --ncpc=2 \
    --model='mnist_fully_connected_IN' \
    --mode='CDP' \
    --round=60 \
    --epsilon=2 \
    --sr=1 \
    --lr=5e-3 \
    --flr=1e-2 \
    --physical_bs=64 \
    --E=1
```

## Step 5: Submit Job

```bash
sbatch run_experiment.sbatch
```

## Step 6: Monitor Job

```bash
# Check job status
squeue -u $USER

# View output
tail -f slurm-<job_id>.out

# View errors
tail -f slurm-<job_id>.err
```

## Example SLURM Scripts

### Multi-GPU Training

```bash
#!/bin/bash
#SBATCH --job-name=privatefl_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=gpu

module load PrgEnv-gnu
module load conda
conda activate privatefl

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $SLURM_SUBMIT_DIR

# Run with multiple GPUs (if your code supports it)
python FedAverage.py --data='cifar10' --model='resnet18_IN' ...
```

### Comparison Job

```bash
#!/bin/bash
#SBATCH --job-name=compare_methods
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=gpu

module load PrgEnv-gnu
module load conda
conda activate privatefl

cd $SLURM_SUBMIT_DIR

python compare_methods.py \
    --data=mnist \
    --mode=CDP \
    --epsilon=2 \
    --nclient=100 \
    --round=60
```

## Tips for Frontier

1. **Use scratch space**: Store large datasets in `$SCRATCH` for faster I/O
2. **Batch jobs**: Submit multiple experiments in a loop
3. **Checkpoint**: Save model checkpoints regularly
4. **Monitor resources**: Use `htop` or `nvidia-smi` to monitor usage
5. **Time limits**: Set appropriate time limits based on your experiment duration

## Troubleshooting

### CUDA Out of Memory
- Reduce `--physical_bs` parameter
- Use smaller batch sizes
- Reduce number of clients per GPU

### Module Not Found
- Ensure conda environment is activated
- Check that all dependencies are installed
- Verify Python path: `which python`

### SLURM Errors
- Check account: `sacctmgr show assoc user=$USER`
- Verify partition availability: `sinfo`
- Check job limits: `sacct -u $USER`

## Example Batch Submission Script

Create `submit_batch.sh`:

```bash
#!/bin/bash

# Array of experiments
experiments=(
    "mnist mnist_fully_connected_IN CDP 2"
    "mnist mnist_fully_connected_MIC CDP 2"
    "cifar10 resnet18_IN CDP 8"
    "cifar10 resnet18_MIC CDP 8"
)

for exp in "${experiments[@]}"; do
    read -r data model mode epsilon <<< "$exp"
    
    sbatch --job-name="${data}_${model}" \
           --output="logs/${data}_${model}_%j.out" \
           --error="logs/${data}_${model}_%j.err" \
           run_experiment.sbatch \
           --data="$data" \
           --model="$model" \
           --mode="$mode" \
           --epsilon="$epsilon"
    
    sleep 1  # Small delay between submissions
done
```

## Contact

For Frontier-specific issues, contact OLCF support: help@olcf.ornl.gov

