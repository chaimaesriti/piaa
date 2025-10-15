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
- Features with **high cardinality** (>90% unique values / total rows by default)
  - Example: In 1000-row dataset, removes features with >900 unique values
  - Adapts to dataset size automatically
- Features with **zero variance** (constant values)

**All thresholds are configurable**

### 3. Feature Selection (`src/features/feature_selection.py`)

**Ranks and selects best features for modeling:**

**Selection Methods:**
- **Mutual Information**: Measures dependency between features and target
- **Tree-based Importance**: Random Forest feature importance
- **Correlation**: Spearman correlation with target
- **Statistical Tests**: ANOVA F-test (classification) or F-regression (regression)

**Selection Criteria:**
- Top K features (e.g., top 10, top 20)
- Threshold-based (score > threshold)
- Ensemble: Aggregates scores across multiple methods

**Redundancy Removal** (NEW):
- Automatically removes highly correlated features (>95% by default)
- Keeps only the best version of each feature
- Example: Selects `BMI` but removes `BMI_capped`, `BMI_binned_10`, `BMI_binned_20`
- Ensures diverse, non-redundant feature set
- Configurable with `--select-max-corr` (default: 0.95)

**Output:**
- Ranked feature list with scores
- Selected features for modeling
- List of redundant features removed
- Detailed feature importance summary

## Usage

### Command Line

```bash
# Basic transformation (auto-saves to transformed_train.csv)
python transform_data.py train.csv

# With target, filtering, and feature selection
python transform_data.py train.csv --target label --filter --select --select-top-k 10

# Full pipeline: transform → filter → select
python transform_data.py train.csv \
  --target label \
  --filter \
  --select \
  --select-top-k 20 \
  --task classification \
  --show-summary

# Custom output or disable auto-save
python transform_data.py train.csv --output custom.csv  # Custom filename
python transform_data.py train.csv --no-save            # Preview only
```

**Auto-save:** Transformed data automatically saved as `transformed_{input_name}.csv`

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

# Test binary feature detection
python tests/test_binary_features.py

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
✅ Feature Selection (ranking & selection for modeling)
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
