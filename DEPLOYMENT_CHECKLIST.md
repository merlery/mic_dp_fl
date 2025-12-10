# Deployment Checklist

Use this checklist to ensure everything is ready for GitHub and Frontier deployment.

## Pre-Deployment

- [ ] Review and update README.md
- [ ] Verify .gitignore excludes large files
- [ ] Test code locally
- [ ] Run comprehensive_test.py to verify everything works
- [ ] Document any special requirements

## GitHub Preparation

- [ ] Initialize git repository (`git init`)
- [ ] Add all files (`git add .`)
- [ ] Create initial commit
- [ ] Create GitHub repository
- [ ] Add remote origin
- [ ] Push to GitHub
- [ ] Verify files on GitHub
- [ ] Check README displays correctly

## Frontier Preparation

- [ ] SSH to Frontier
- [ ] Clone repository
- [ ] Run `script/frontier_setup.sh`
- [ ] Verify environment activation
- [ ] Test imports: `python -c "import torch; print('OK')"`
- [ ] Create SLURM script
- [ ] Test with small job
- [ ] Verify output files

## Code Verification

- [ ] All Python files have proper imports
- [ ] No hardcoded paths
- [ ] Relative paths used where possible
- [ ] Error handling in place
- [ ] Logging configured

## Documentation

- [ ] README.md complete
- [ ] FRONTIER_SETUP.md complete
- [ ] GITHUB_DEPLOYMENT.md complete
- [ ] Code comments adequate
- [ ] Parameter descriptions clear

## Files to Include

- [x] Core Python files
- [x] Model definitions
- [x] Utility scripts
- [x] Setup scripts
- [x] Documentation
- [x] Requirements.txt
- [x] .gitignore

## Files to Exclude (via .gitignore)

- [x] __pycache__/
- [x] *.pyc
- [x] Large data files (*.npy, *.tif)
- [x] Model checkpoints (*.pth, *.pt)
- [x] Log files
- [x] Virtual environments
- [x] IDE files

## Final Steps

- [ ] Test clone on a fresh machine
- [ ] Verify setup script works
- [ ] Test a small experiment
- [ ] Document any issues found
- [ ] Update documentation if needed

## Quick Test Commands

```bash
# Test imports
python -c "from modelUtil import *; print('Imports OK')"

# Test MIC utils
python -c "from mic_utils import compute_mic_matrix; print('MIC OK')"

# Test model creation
python -c "from modelUtil import mnist_fully_connected_MIC; m=mnist_fully_connected_MIC(10); print('Model OK')"

# Run comprehensive test
python comprehensive_test.py
```

## Ready for Deployment?

If all items are checked, you're ready to:
1. Push to GitHub
2. Clone on Frontier
3. Run experiments!

