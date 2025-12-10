# Deployment Summary - Step by Step Instructions

## 📋 Overview

This document provides complete step-by-step instructions for:
1. Setting up GitHub repository
2. Deploying to ORNL Frontier
3. Running experiments

## 🚀 Part 1: GitHub Setup

### Step 1.1: Initialize Git Repository

```bash
# Navigate to project directory
cd /path/to/usenix2026

# Initialize git (if not already done)
git init

# Check what will be committed
git status

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PrivateFL with MIC implementation

- Baseline (IN) and MIC-based transformation methods
- Support for multiple datasets and models
- Differential privacy (CDP/LDP) support
- Ready for ORNL Frontier deployment"
```

### Step 1.2: Create GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `privatefl` (or your choice)
3. **Description**: "PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation"
4. **Visibility**: Choose Public or Private
5. **Important**: Do NOT check "Initialize with README" (we already have one)
6. **Click**: "Create repository"

### Step 1.3: Connect and Push

```bash
# Add remote (replace <username> and <repo-name>)
git remote add origin https://github.com/<username>/<repo-name>.git

# Verify remote
git remote -v

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 1.4: Verify on GitHub

1. Visit your repository: `https://github.com/<username>/<repo-name>`
2. Verify files are present
3. Check README.md displays correctly
4. Verify .gitignore is working (no large files uploaded)

## 🖥️ Part 2: ORNL Frontier Setup

### Step 2.1: SSH to Frontier

```bash
ssh username@frontier.olcf.ornl.gov
```

### Step 2.2: Navigate to Work Directory

```bash
# Check your project directory
echo $PROJWORK

# Navigate (replace <project_id> with your actual project ID)
cd $PROJWORK/<project_id>

# Or use scratch space for temporary files
# cd $SCRATCH
```

### Step 2.3: Clone Repository

```bash
# Clone using HTTPS
git clone https://github.com/<username>/<repo-name>.git

# Or using SSH (if configured)
# git clone git@github.com:<username>/<repo-name>.git

# Navigate into directory
cd <repo-name>
```

### Step 2.4: Set Up Environment

**Option A: Using Setup Script (Recommended)**

```bash
# Make script executable
chmod +x script/frontier_setup.sh

# Run setup
bash script/frontier_setup.sh
```

**Option B: Manual Setup**

```bash
# Load modules
module load PrgEnv-gnu
module load conda

# Create environment
conda create --name privatefl python=3.8 -y
conda activate privatefl

# Install PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
pip install -r requirements.txt

# Fix opacus if needed
pip install opacus==1.0.0
```

### Step 2.5: Verify Installation

```bash
# Activate environment
conda activate privatefl

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from modelUtil import *; print('Models OK')"
python -c "from mic_utils import compute_mic_matrix; print('MIC OK')"

# Run comprehensive test
python comprehensive_test.py
```

## 📝 Part 3: Create and Submit Jobs

### Step 3.1: Prepare SLURM Script

```bash
# Copy example script
cp script/frontier_run_example.sbatch my_experiment.sbatch

# Edit script
nano my_experiment.sbatch
# Or: vi my_experiment.sbatch
```

**Update these lines in the script:**
```bash
#SBATCH --account=<YOUR_ACCOUNT>  # Replace with your account
# Update experiment parameters as needed
```

### Step 3.2: Submit Job

```bash
# Submit job
sbatch my_experiment.sbatch

# Note the job ID (e.g., 12345678)
```

### Step 3.3: Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (replace <job_id> with actual ID)
tail -f slurm-<job_id>.out

# View errors
tail -f slurm-<job_id>.err

# Check job details
scontrol show job <job_id>
```

### Step 3.4: Check Results

```bash
# Results are saved in log/ directory
ls -lh log/

# View results
cat log/E1/mnist_100_2_CDP_*.csv
```

## 🔄 Part 4: Running Multiple Experiments

### Option A: Submit Multiple Jobs

```bash
# Create a loop script
cat > submit_all.sh << 'EOF'
#!/bin/bash
for epsilon in 2 4 8; do
    for model in mnist_fully_connected_IN mnist_fully_connected_MIC; do
        sbatch --job-name="${model}_eps${epsilon}" \
               --output="logs/${model}_eps${epsilon}_%j.out" \
               script/frontier_run_example.sbatch \
               --data=mnist \
               --model=$model \
               --epsilon=$epsilon
        sleep 1
    done
done
EOF

chmod +x submit_all.sh
./submit_all.sh
```

### Option B: Use Job Arrays

```bash
# Create array script (see SLURM documentation)
#SBATCH --array=1-10
```

## 📊 Part 5: Example Experiments

### Baseline (IN) Method

```bash
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

### MIC-Based Method

```bash
python FedAverage.py \
    --data='mnist' \
    --nclient=100 \
    --nclass=10 \
    --ncpc=2 \
    --model='mnist_fully_connected_MIC' \
    --mode='CDP' \
    --round=60 \
    --epsilon=2 \
    --sr=1 \
    --lr=5e-3 \
    --flr=1e-2 \
    --physical_bs=64 \
    --E=1
```

### Compare Both Methods

```bash
python compare_methods.py \
    --data=mnist \
    --mode=CDP \
    --epsilon=2 \
    --nclient=100 \
    --round=60
```

## 🛠️ Troubleshooting

### Git Issues

```bash
# If push fails
git remote -v  # Check remote URL
git remote set-url origin <correct-url>

# If large files error
git rm --cached <large-file>
git commit -m "Remove large file"
```

### Frontier Issues

```bash
# Module not found
module avail python
module load python/3.8-anaconda3

# CUDA not found
module avail cuda
module load cuda

# Environment not activating
source ~/.bashrc
conda init bash
```

### Job Issues

```bash
# Job pending too long
squeue -u $USER
# Check partition: sinfo
# Check account: sacctmgr show assoc user=$USER

# Job failed
cat slurm-<job_id>.err
# Check common issues: CUDA OOM, missing modules, etc.
```

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] Repository pushed to GitHub
- [ ] Repository cloned on Frontier
- [ ] Environment set up and activated
- [ ] Imports work: `python -c "import torch; from modelUtil import *"`
- [ ] Test passed: `python comprehensive_test.py`
- [ ] SLURM script created and edited
- [ ] Account name updated in SLURM script
- [ ] Ready to submit jobs!

## 📚 Additional Resources

- **Frontier Documentation**: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
- **SLURM Guide**: https://slurm.schedmd.com/documentation.html
- **GitHub Help**: https://docs.github.com

## 🎯 Quick Reference

```bash
# GitHub
git add . && git commit -m "message" && git push

# Frontier
ssh username@frontier.olcf.ornl.gov
cd $PROJWORK/<project_id>
git pull
conda activate privatefl
sbatch my_job.sbatch
squeue -u $USER
```

## 📞 Support

- **GitHub Issues**: Create issue in repository
- **Frontier Support**: help@olcf.ornl.gov
- **Project Questions**: Check documentation files

---

**You're all set!** Follow the steps above to deploy and run experiments. Good luck! 🚀

