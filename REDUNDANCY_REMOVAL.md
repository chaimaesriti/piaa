# Redundancy Removal in Feature Selection

## Problem

When using feature transformations (capping, binning), you create multiple correlated features from the same base feature. For example:

```
BMI → BMI_capped, BMI_binned_10, BMI_binned_20
```

All these features are highly correlated (>95%). Selecting multiple variants wastes feature slots and provides no additional information to the model.

## Solution

**Redundancy removal** automatically detects and removes highly correlated features, keeping only the best version of each.

## How It Works

1. **Rank features** by aggregated score (mutual info + tree importance + correlation)
2. **Iterate through ranked features**:
   - Check correlation with already-selected features
   - If correlation > threshold (default 0.95), skip the feature
   - Otherwise, add to selected set
3. **Result**: Keep the highest-scoring version of each feature

## Usage

### Default Behavior (Recommended)

```bash
# Default: max_correlation=0.95 (remove features correlated >95%)
python transform_data.py train.csv --target Response --select --select-top-k 20
```

### Custom Correlation Threshold

```bash
# Stricter: Remove features correlated >90%
python transform_data.py train.csv --target Response --select --select-top-k 20 --select-max-corr 0.90

# More lenient: Remove features correlated >98%
python transform_data.py train.csv --target Response --select --select-top-k 20 --select-max-corr 0.98

# Disable redundancy removal
python transform_data.py train.csv --target Response --select --select-top-k 20 --select-max-corr 1.0
```

### Full Pipeline with Filtering

```bash
python transform_data.py train.csv \
  --target Response \
  --filter \
  --select \
  --select-top-k 30 \
  --select-max-corr 0.95
```

## Example Output

### Before (Without Redundancy Removal)

```
Top 20 Features:
  1. BMI                    (score: 0.2292)
  2. BMI_capped             (score: 0.2288)  ← Redundant!
  3. BMI_binned_20          (score: 0.2167)  ← Redundant!
  4. BMI_binned_10          (score: 0.2095)  ← Redundant!
  5. Wt                     (score: 0.1899)
  6. Wt_capped              (score: 0.1893)  ← Redundant!
  7. Wt_binned_20           (score: 0.1828)  ← Redundant!
  8. Wt_binned_10           (score: 0.1727)  ← Redundant!
  ...
```

**Problem**: 8 feature slots used for just 2 base features (BMI, Wt)!

### After (With Redundancy Removal)

```
Top 20 Features:
  1. BMI                    (score: 0.2292)  ✓ Selected
  2. BMI_capped             (score: 0.2288)  ✗ Redundant with BMI (corr: 0.998)
  3. BMI_binned_20          (score: 0.2167)  ✗ Redundant with BMI (corr: 0.985)
  4. BMI_binned_10          (score: 0.2095)  ✗ Redundant with BMI (corr: 0.972)
  5. Wt                     (score: 0.1899)  ✓ Selected
  6. Wt_capped              (score: 0.1893)  ✗ Redundant with Wt (corr: 0.997)
  7. Wt_binned_20           (score: 0.1828)  ✗ Redundant with Wt (corr: 0.981)
  8. Wt_binned_10           (score: 0.1727)  ✗ Redundant with Wt (corr: 0.963)
  9. Medical_History_23     (score: 0.1304)  ✓ Selected
 10. Medical_Keyword_15     (score: 0.1244)  ✓ Selected
 ...

Redundant Features (removed due to high correlation):
  BMI_capped          → correlated with BMI (0.998)
  BMI_binned_20       → correlated with BMI (0.985)
  BMI_binned_10       → correlated with BMI (0.972)
  Wt_capped           → correlated with Wt (0.997)
  Wt_binned_20        → correlated with Wt (0.981)
  Wt_binned_10        → correlated with Wt (0.963)
```

**Result**: 20 diverse, non-redundant features selected!

## Recommended Thresholds

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.90 | Strict | Remove any features with >90% correlation |
| **0.95** | **Default** | **Remove features with >95% correlation (recommended)** |
| 0.98 | Lenient | Only remove very highly correlated features |
| 1.00 | Disabled | No redundancy removal |

## For Your 279-Column Dataset

```bash
# Select top 30 diverse features (no redundant BMI/Wt/Medical_History variants)
python transform_data.py train.csv --target Response --select --select-top-k 30 --select-max-corr 0.95

# More aggressive: Top 20 with stricter correlation threshold
python transform_data.py train.csv --target Response --select --select-top-k 20 --select-max-corr 0.90

# Conservative: Top 50 with lenient threshold
python transform_data.py train.csv --target Response --select --select-top-k 50 --select-max-corr 0.98
```

## Python API

```python
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig

# Configure with redundancy removal
config = FeatureSelectionConfig(
    methods=['mutual_info', 'tree_importance', 'correlation'],
    top_k=30,
    max_correlation=0.95,  # Remove features correlated >95%
    task='classification'
)

fs = FeatureSelector(config)
fs.fit(X, y)
fs.print_summary(top_n=30)

# Get selected features
X_selected = fs.transform(X)

# Check which features were removed as redundant
print(f"Removed as redundant: {fs.removed_redundant}")
```

## Benefits

✅ **No Redundancy**: Each selected feature provides unique information
✅ **Better Models**: Diverse features → better model generalization
✅ **Faster Training**: Fewer correlated features → less multicollinearity
✅ **Easier Interpretation**: One version of each feature (not 4 variants)
✅ **Automatic**: Picks the best version of each feature automatically

## Testing

The redundancy removal is fully tested:

```bash
python tests/test_redundancy_removal.py
```

Expected output:
```
✓ Selected exactly 3 features
✓ Removed 2 redundant features
✓ Only 1 age variant selected (avoiding redundancy)
✓ Selected diverse features: ['age', 'income', 'score']
```

## Summary

**Before**: BMI, BMI_capped, BMI_binned_10, BMI_binned_20 (4 slots for 1 feature)
**After**: BMI (1 slot, best version automatically chosen)

This gives you **diverse, non-redundant features** for better model performance.
