# Installing minepy for MIC Support

## Issue
The `minepy` library is required for MIC (Maximum Information Coefficient) computation but may not be installed by default.

## Solution

### Option 1: Install minepy directly
```bash
pip install minepy
```

### Option 2: Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Option 3: If pip install fails (Windows issues)

**For Windows users**, `minepy` might require compilation. Try:

1. **Install pre-built wheel** (if available):
   ```bash
   pip install --upgrade pip
   pip install minepy
   ```

2. **Use conda** (if using conda environment):
   ```bash
   conda install -c conda-forge minepy
   ```

3. **Install build tools** (if compilation fails):
   - Install Microsoft C++ Build Tools
   - Or install Visual Studio with C++ support
   - Then retry: `pip install minepy`

### Option 4: Alternative - Use without MIC

If `minepy` installation fails, you can still use the code:
- The baseline (IN) models work without minepy
- MIC models can be created but won't compute MIC values
- Models will use default initialization instead of MIC-based initialization

## Verification

After installation, verify with:
```bash
python quick_test.py
```

Or test minepy directly:
```python
from minepy import MINE
print("minepy installed successfully!")
```

## Troubleshooting

### Error: "No module named 'minepy'"
- Make sure you're in the correct conda/virtual environment
- Try: `conda activate privatefl` (or your environment name)
- Then: `pip install minepy`

### Error: "Microsoft Visual C++ 14.0 is required"
- Install Visual Studio Build Tools
- Or use conda: `conda install -c conda-forge minepy`

### Error: "Failed building wheel for minepy"
- Try: `pip install --upgrade pip setuptools wheel`
- Then: `pip install minepy`
- Or use conda as above

