# Feature Selection Guide

## Overview

Feature selection ranks features by importance and selects the best ones for modeling, helping to:
- Reduce dimensionality
- Improve model performance
- Reduce training time
- Reduce overfitting
- Improve interpretability

## Selection Methods

### 1. Mutual Information
**Measures**: Non-linear dependency between feature and target
**Best for**: Both numerical and categorical features
**Captures**: Complex relationships that correlation might miss

### 2. Random Forest Importance
**Measures**: How much each feature decreases impurity in decision trees
**Best for**: Non-linear relationships, feature interactions
**Captures**: Feature importance in tree-based models

### 3. Correlation (Spearman)
**Measures**: Monotonic relationship with target
**Best for**: Quick analysis, linear relationships
**Captures**: Rank-based correlation (robust to outliers)

### 4. Statistical Tests
**Measures**: Statistical significance of relationship
**Uses**: ANOVA F-test (classification) or F-regression (regression)
**Best for**: Understanding statistical relevance

## Usage

### Basic Usage

```bash
# Select top 10 features
python transform_data.py train.csv --target label --select --select-top-k 10
```

### Selection Criteria

#### Option 1: Top K Features
```bash
# Select top 15 features
python transform_data.py train.csv --target label --select --select-top-k 15
```
Best for: When you know how many features you want

#### Option 2: Threshold-based
```bash
# Select features with score > 0.1
python transform_data.py train.csv --target label --select --select-threshold 0.1
```
Best for: When you want features above certain quality threshold

### Method Selection

```bash
# Use all methods (ensemble - recommended)
python transform_data.py train.csv --target label --select \
  --select-methods mutual_info tree_importance correlation statistical \
  --select-top-k 20

# Use single method
python transform_data.py train.csv --target label --select \
  --select-methods mutual_info \
  --select-top-k 10

# Use combination
python transform_data.py train.csv --target label --select \
  --select-methods mutual_info correlation \
  --select-top-k 15
```

### Task Type

```bash
# Classification (default)
python transform_data.py train.csv --target label --select --task classification

# Regression
python transform_data.py train.csv --target price --select --task regression
```

## Full Pipeline

### Recommended Workflow

```bash
# Step 1: Transform + Filter + Select
python transform_data.py train.csv \
  --target label \
  --filter \
  --select \
  --select-top-k 20 \
  --task classification
```

This runs:
1. **Transform**: Creates capped, binned, grouped features
2. **Filter**: Removes high missingness, high cardinality, zero variance
3. **Select**: Ranks remaining features and selects top 20

### Output

The command shows:

```
============================================================
FEATURE SELECTION
============================================================
Task: classification
Features: 50
Samples: 1000
Methods: mutual_info, tree_importance, correlation

Computing mutual_info scores...
  ✓ Computed mutual_info scores

Computing tree_importance scores...
  ✓ Computed tree_importance scores

Computing correlation scores...
  ✓ Computed correlation scores

============================================================
FEATURE SELECTION CRITERIA
============================================================
Criterion: Top 20 features
Selected: 20 / 50 features

============================================================
FEATURE SELECTION SUMMARY
============================================================

Total features evaluated: 50
Features selected: 20
Selection rate: 40.0%

============================================================
Top 20 Features (by aggregated score):
============================================================
        feature  aggregated_score  selected  mutual_info_score  ...
     feature_A            0.8523      True             0.9234  ...
     feature_B            0.7891      True             0.8123  ...
     feature_C            0.7456      True             0.7654  ...
     ...
```

## Python API

```python
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig

# Configure selector
config = FeatureSelectionConfig(
    methods=['mutual_info', 'tree_importance', 'correlation'],
    top_k=15,
    task='classification'
)

# Fit and select
fs = FeatureSelector(config)
X_selected = fs.fit_transform(X, y)

# Get selected features
selected_features = fs.get_selected_features()
print(f"Selected: {selected_features}")

# Get feature scores
scores = fs.get_feature_scores()  # Aggregated scores
mi_scores = fs.get_feature_scores('mutual_info')  # Method-specific

# Get top features
top_10 = fs.get_top_features(k=10)

# Print detailed summary
fs.print_summary(top_n=20)
```

## Tips & Best Practices

### 1. Use Multiple Methods
**Recommended**: Use at least 2-3 methods
```bash
--select-methods mutual_info tree_importance correlation
```
This creates an ensemble that's more robust than single methods.

### 2. Start Conservative
Start with more features, then reduce:
```bash
# First run: Keep top 30
python transform_data.py train.csv --target label --select --select-top-k 30

# If overfitting: Reduce to top 15
python transform_data.py train.csv --target label --select --select-top-k 15
```

### 3. Check Feature Importance
Always review the feature importance summary to understand:
- Which features are most important
- If any expected features are missing
- If any surprising features rank high

### 4. Task-Specific Selection
- **Classification**: Use all methods (mutual_info, tree_importance, correlation, statistical)
- **Regression**: Focus on correlation and tree_importance

### 5. Combine with Filtering
Always use filtering before selection:
```bash
python transform_data.py train.csv --target label --filter --select --select-top-k 20
```
This removes garbage features before ranking.

## Examples

### Example 1: Small Dataset (few features)
```bash
# Don't over-select on small datasets
python transform_data.py small.csv --target label --select --select-top-k 5
```

### Example 2: Large Dataset (many features)
```bash
# Aggressive selection for many features
python transform_data.py large.csv --target label \
  --filter \
  --select \
  --select-top-k 50 \
  --select-methods mutual_info tree_importance
```

### Example 3: Regression Task
```bash
python transform_data.py house_prices.csv --target price \
  --select \
  --task regression \
  --select-top-k 20
```

### Example 4: Quick Analysis (correlation only)
```bash
# Fast feature ranking
python transform_data.py train.csv --target label \
  --select \
  --select-methods correlation \
  --select-top-k 15
```

## Troubleshooting

### Error: "Target is required for feature selection"
**Solution**: Add `--target your_target_column`

### Warning: "Small sample size may affect tree_importance"
**Solution**: With <100 samples, use `--select-methods mutual_info correlation`

### Features ranked unexpectedly
**Possible causes**:
1. Target leakage (feature derived from target)
2. High correlation between features
3. Feature needs different encoding

**Solution**: Review features manually, check feature engineering

## Performance

| Features | Samples | Time (mutual_info) | Time (tree_importance) | Time (all methods) |
|----------|---------|-------------------|----------------------|-------------------|
| 50       | 1,000   | ~1s              | ~3s                  | ~5s              |
| 100      | 10,000  | ~3s              | ~15s                 | ~20s             |
| 500      | 10,000  | ~10s             | ~45s                 | ~60s             |

## Next Steps

After feature selection:
1. **Save selected features**: Output is auto-saved to `transformed_{input}.csv`
2. **Train models**: Use selected features for modeling
3. **Iterate**: If performance is poor, adjust selection criteria
4. **Track experiments**: Record which feature set works best
