# PIAA Feature Engineering & Quality Filtering

Complete pipeline for feature transformation and quality filtering before modeling.

## What's Implemented

### 1. Feature Engineering (`src/features/feature_engineering.py`)

**Numerical Features** - Creates 3 variants for each:
- `feature_capped`: Outliers capped at percentiles (default: 1st/99th)
- `feature_binned_10`: Discretized into 10 bins
- `feature_binned_20`: Discretized into 20 bins

**Categorical Features**:
- Groups rare categories (< threshold frequency) together
- Creates: `category_grouped` with combined labels like `cat1_cat2_cat3_other`

**Binary Features** (automatically detected):
- Features with exactly 2 unique values (e.g., 0/1, True/False, M/F)
- **Kept as-is without transformation**
- No capping, binning, or grouping applied

**Target Column**:
- Can be specified with `target_col` parameter
- **Excluded from all transformations**
- Preserved in final dataset unchanged

### 2. Feature Quality Filter (`src/features/feature_filter.py`)

**Automatically removes:**
- Features with **high missingness** (>90% by default)
- Features with **high cardinality**:
  - Numerical: ≥1000 unique values
  - Categorical: ≥100 unique values
- Features with **zero variance** (constant values)

**All thresholds are configurable**

## Usage

### Command Line

```bash
# Basic transformation
python transform_data.py train.csv

# With quality filtering
python transform_data.py train.csv --filter

# Full pipeline with options
python transform_data.py train.csv \
  --filter \
  --max-missing 0.80 \
  --max-cardinality-num 500 \
  --max-cardinality-cat 50 \
  --target label \
  --output train_transformed.csv \
  --show-summary
```

See `USAGE.md` for all options.

### Python API

```python
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig

# 1. Transform features (binary features auto-detected, target excluded)
fe = FeatureEngineer()
df_transformed = fe.fit_transform_numerical(df, numerical_cols, target_col='label')
df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols, target_col='label')

# Check what was detected
print(f"Binary features: {fe.binary_features}")  # e.g., ['is_active', 'has_loan']
print(f"Target: {fe.target_col}")  # 'label'

# 2. Filter low-quality features
ff = FeatureFilter()
df_filtered = ff.fit_transform(df_transformed, target_col='label')

# 3. Get model-ready features
features = [col for col in df_filtered.columns if col != 'label']
X = df_filtered[features]
y = df_filtered['label']
```

## Tests

```bash
# Test feature engineering
python tests/test_feature_engineering.py

# Test quality filtering
python tests/test_feature_filter.py

# Test full pipeline
python tests/test_full_pipeline.py
```

## Project Structure

```
piaa/
├── src/
│   ├── features/
│   │   ├── feature_engineering.py  # Transform features
│   │   └── feature_filter.py       # Quality filtering
│   ├── models/                     # (Next: TabNet, MLP, NN)
│   └── utils/
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_feature_filter.py
│   └── test_full_pipeline.py
├── data/                           # Put your train.csv here
├── transform_data.py               # CLI tool
├── USAGE.md                        # Detailed usage guide
└── README.md                       # This file
```

## What's Next

✅ Feature Engineering (capped, binned, grouped)
✅ Quality Filtering (missingness, cardinality, variance)
⬜ Feature Set Registry (track which features used)
⬜ Models (TabNet, MLP, NN with softmax)
⬜ Experiment Tracking (features + model + metrics)

## Example Output

```
============================================================
FEATURE FILTER SUMMARY
============================================================
Total features:            20
Kept features:             18 (90.0%)
Removed features:          2

Removal reasons:
  - High missingness (>90%): 2
  - High cardinality:        0
  - Zero variance:           0
============================================================
```

## Key Design Principles

1. **Consistent transformations**: Fit on train, transform on test with same parameters
2. **Configurable thresholds**: All filtering criteria can be adjusted
3. **Tracking**: All transformations stored for reproducibility
4. **Modular**: Each step can be used independently
