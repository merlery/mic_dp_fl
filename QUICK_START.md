# Quick Start Guide

## For GitHub Deployment

### Step 1: Initialize Git (if not done)
```bash
git init
git add .
git commit -m "Initial commit: PrivateFL with MIC implementation"
```

### Step 2: Create GitHub Repository
1. Go to https://github.com/new
2. Create repository (don't initialize with README)
3. Copy the repository URL

### Step 3: Connect and Push
```bash
git remote add origin https://github.com/<username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## For ORNL Frontier

### Step 1: SSH to Frontier
```bash
ssh username@frontier.olcf.ornl.gov
```

### Step 2: Clone Repository
```bash
cd $PROJWORK/<your_project_id>
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### Step 3: Setup Environment
```bash
bash script/frontier_setup.sh
```

### Step 4: Create and Submit Job
```bash
# Copy example script
cp script/frontier_run_example.sbatch my_job.sbatch

# Edit script (update account name)
nano my_job.sbatch

# Submit job
sbatch my_job.sbatch

# Monitor
squeue -u $USER
tail -f slurm-<job_id>.out
```

## That's It!

See detailed guides:
- [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md) - Full GitHub setup
- [FRONTIER_SETUP.md](FRONTIER_SETUP.md) - Full Frontier setup
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Verification checklist

