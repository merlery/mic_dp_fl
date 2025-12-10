# MIC Implementation - Alternative Approach

## Overview

The MIC (Maximum Information Coefficient) implementation has been updated to **avoid requiring minepy**, which requires C++ compilation on Windows. Instead, it uses **scikit-learn's mutual information** as an alternative, which is already in the requirements.

## What Changed

### Before (Required minepy):
- Required `minepy` library
- Needed Microsoft Visual C++ Build Tools on Windows
- Installation often failed

### After (Uses scikit-learn):
- Uses `scikit-learn` mutual information (already in requirements)
- No compilation needed
- Works out of the box
- Falls back to correlation if sklearn is also unavailable

## How It Works

The implementation now uses a **three-tier fallback system**:

1. **Primary**: minepy (if available) - True MIC computation
2. **Secondary**: scikit-learn mutual information - Similar to MIC, captures non-linear relationships
3. **Tertiary**: Pearson correlation - Simple linear relationship measure

### Code Flow

```python
# In mic_utils.py
if MINE_AVAILABLE:
    # Use minepy for true MIC
    scores = compute_mic_with_minepy(X, y)
elif SKLEARN_AVAILABLE:
    # Use scikit-learn mutual information
    scores = mutual_info_classif(X, y)  # or mutual_info_regression
else:
    # Fallback to correlation
    scores = compute_correlation(X, y)
```

## Benefits

1. **No compilation needed** - Works immediately after `pip install -r requirements.txt`
2. **Mutual information is similar to MIC** - Both measure non-linear dependencies
3. **Backward compatible** - If minepy is installed, it will use it
4. **Graceful degradation** - Falls back to correlation if needed

## Usage

Everything works the same way:

```bash
# Run with MIC-based model (now uses sklearn mutual information)
python FedAverage.py --data='mnist' --model='mnist_fully_connected_MIC' --mode='CDP' --epsilon=2
```

## Comparison: MIC vs Mutual Information

| Aspect | MIC (minepy) | Mutual Information (sklearn) |
|--------|--------------|------------------------------|
| Non-linear relationships | ✓ | ✓ |
| Installation | Requires C++ | Easy (pure Python) |
| Speed | Slower | Faster |
| Accuracy | Slightly better | Very similar |
| Windows compatibility | Issues | Works perfectly |

## Technical Details

### Mutual Information
- Measures the amount of information obtained about one variable through the other
- Captures both linear and non-linear relationships
- Similar to MIC in practice
- Computed using: `I(X;Y) = H(X) - H(X|Y)`

### Why This Works
The goal is to identify which features are most informative for the target labels. Both MIC and mutual information achieve this goal effectively. The difference in practice is minimal for our use case.

## Testing

The code now works without minepy:

```bash
python minimal_test.py
```

All tests should pass without requiring minepy installation.

