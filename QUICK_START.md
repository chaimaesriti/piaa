# Quick Start Guide

## Test with Your Data

### 1. Basic transformation
```bash
python transform_data.py train.csv --target your_target_column
```
**Saves to:** `transformed_train.csv`

What happens:
- ✓ Binary features (2 values) auto-detected and kept as-is
- ✓ Target column excluded from transformations
- ✓ Numerical features: capped + binned (10, 20)
- ✓ Categorical features: rare categories grouped
- ✓ **Automatically saved** with `transformed_` prefix

### 2. With quality filtering (recommended)
```bash
python transform_data.py train.csv --target your_target_column --filter
```
**Saves to:** `transformed_train.csv`

Additional filters:
- ✓ Removes features with >90% missing values
- ✓ Removes high cardinality features (>90% unique/rows - adapts to dataset size)
- ✓ Removes zero-variance features
- ✓ Saves clean, model-ready data

### 2a. With feature selection (best for modeling)
```bash
python transform_data.py train.csv --target label --filter --select --select-top-k 15
```
**Saves to:** `transformed_train.csv`

Complete pipeline:
- ✓ Transforms features (capping, binning, grouping)
- ✓ Filters low-quality features
- ✓ **Ranks and selects top 15 features** using:
  - Mutual Information
  - Random Forest importance
  - Correlation analysis
- ✓ Ready for model training!

### 3. Custom output or preview only
```bash
# Custom filename
python transform_data.py train.csv --target label --output my_output.csv

# Preview only (no save)
python transform_data.py train.csv --target label --no-save
```

### 4. Custom thresholds
```bash
python transform_data.py train.csv \
  --target label \
  --filter \
  --max-missing 0.80 \
  --max-cardinality-num 500 \
  --max-cardinality-cat 50 \
  --bins 10 20 50 \
  --show-summary
```

## What You'll See

```
============================================================
TRANSFORMING FEATURES
============================================================

Transforming 5 numerical features...

ℹ  Binary features detected (skipping transformation): ['is_active', 'has_loan']
ℹ  Target column skipped: target

✓ Created numerical transformations

============================================================
FILTERING FEATURES
============================================================

============================================================
FEATURE FILTER SUMMARY
============================================================
Total features:            20
Kept features:             18 (90.0%)
Removed features:          2

Removal reasons:
  - High missingness (>90%): 1
  - High cardinality:        1
  - Zero variance:           0
============================================================
```

## Run Demo

```bash
python demo_binary_detection.py
```

See how binary features and target columns are handled with example data.

## Run Tests

```bash
# All tests
python tests/test_feature_engineering.py
python tests/test_feature_filter.py
python tests/test_binary_features.py
python tests/test_full_pipeline.py
```

## Python API

```python
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_filter import FeatureFilter

# Transform
fe = FeatureEngineer()
df = fe.fit_transform_numerical(df, num_cols, target_col='target')
df = fe.fit_transform_categorical(df, cat_cols, target_col='target')

# Filter
ff = FeatureFilter()
df = ff.fit_transform(df, target_col='target')

# Get features
X_cols = [c for c in df.columns if c != 'target']
X = df[X_cols]
y = df['target']
```

## Next Steps

See full documentation:
- `README.md` - Complete overview
- `USAGE.md` - All CLI options
- `CHANGELOG.md` - What's new
