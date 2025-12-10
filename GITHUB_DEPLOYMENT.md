# GitHub Deployment Guide

This guide walks you through deploying the PrivateFL project to GitHub and preparing it for execution on ORNL Frontier.

## Step 1: Prepare Repository

### 1.1 Clean Up Test Files (Optional)

You can either:
- **Option A**: Keep test files (recommended for development)
- **Option B**: Move tests to a `tests/` directory
- **Option C**: Remove test files if not needed

```bash
# Option B: Organize tests
mkdir -p tests
mv *_test.py tests/
mv test_*.py tests/
```

### 1.2 Verify .gitignore

Ensure `.gitignore` is in place to exclude:
- Large data files
- Model checkpoints
- Python cache files
- Log files

## Step 2: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PrivateFL with MIC implementation"
```

## Step 3: Create GitHub Repository

### 3.1 On GitHub Website

1. Go to https://github.com
2. Click "New repository"
3. Repository name: `privatefl` (or your preferred name)
4. Description: "PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation"
5. Choose visibility (Public/Private)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 3.2 Connect Local Repository

```bash
# Add remote (replace <username> and <repo-name>)
git remote add origin https://github.com/<username>/<repo-name>.git

# Or using SSH:
# git remote add origin git@github.com:<username>/<repo-name>.git

# Verify remote
git remote -v
```

## Step 4: Push to GitHub

```bash
# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 5: Verify on GitHub

1. Visit your repository on GitHub
2. Verify all files are present
3. Check that README.md displays correctly
4. Verify .gitignore is working (large files should not be uploaded)

## Step 6: Clone on Frontier

### 6.1 SSH to Frontier

```bash
ssh username@frontier.olcf.ornl.gov
```

### 6.2 Navigate to Work Directory

```bash
cd $PROJWORK/<your_project_id>
# Or use scratch: cd $SCRATCH
```

### 6.3 Clone Repository

```bash
# Using HTTPS
git clone https://github.com/<username>/<repo-name>.git

# Or using SSH (if you've set up SSH keys)
# git clone git@github.com:<username>/<repo-name>.git

cd <repo-name>
```

### 6.4 Set Up Environment

```bash
# Run setup script
bash script/frontier_setup.sh

# Or manually:
module load PrgEnv-gnu
module load conda
conda create --name privatefl python=3.8 -y
conda activate privatefl
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Step 7: Run Experiments on Frontier

### 7.1 Prepare SLURM Script

```bash
# Copy example script
cp script/frontier_run_example.sbatch my_experiment.sbatch

# Edit the script
nano my_experiment.sbatch
# Update: --account=<YOUR_ACCOUNT>
# Update: experiment parameters
```

### 7.2 Submit Job

```bash
sbatch my_experiment.sbatch
```

### 7.3 Monitor

```bash
# Check status
squeue -u $USER

# View output
tail -f slurm-<job_id>.out
```

## Step 8: Update Repository (Ongoing)

When making changes:

```bash
# On your local machine
git add .
git commit -m "Description of changes"
git push origin main

# On Frontier
cd <repo-directory>
git pull origin main
```

## Repository Structure Checklist

Ensure your repository includes:

- [x] README.md - Main documentation
- [x] .gitignore - Excludes large files
- [x] requirements.txt - Dependencies
- [x] FRONTIER_SETUP.md - Frontier instructions
- [x] GITHUB_DEPLOYMENT.md - This file
- [x] Core Python files (FedAverage.py, etc.)
- [x] Scripts directory
- [x] Model definitions
- [ ] LICENSE file (add if needed)
- [ ] CONTRIBUTING.md (optional)

## Large Files Handling

If you need to include large files:

1. **Use Git LFS** (for files > 100MB):
   ```bash
   git lfs install
   git lfs track "*.npy"
   git lfs track "*.pth"
   git add .gitattributes
   ```

2. **Or exclude them** (recommended):
   - Keep data files on Frontier's filesystem
   - Download datasets on Frontier
   - Use symbolic links if needed

## Troubleshooting

### Git Push Fails

```bash
# Check remote URL
git remote -v

# Update remote if needed
git remote set-url origin <new-url>
```

### Large Files Error

```bash
# Remove large files from git history (if accidentally added)
git rm --cached <large-file>
git commit -m "Remove large file"
```

### Frontier Clone Issues

```bash
# Use HTTPS if SSH fails
git clone https://github.com/<username>/<repo>.git

# Or configure git credentials
git config --global credential.helper store
```

## Next Steps

1. ✅ Repository is on GitHub
2. ✅ Cloned on Frontier
3. ✅ Environment set up
4. ✅ Ready to run experiments!

See [FRONTIER_SETUP.md](FRONTIER_SETUP.md) for detailed Frontier instructions.

