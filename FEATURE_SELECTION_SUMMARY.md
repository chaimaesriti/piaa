# Feature Selection - Implementation Summary

## ✅ What Was Added

### Feature Selection Module (`src/features/feature_selection.py`)

Complete feature ranking and selection system with **4 selection methods**:

1. **Mutual Information** - Measures non-linear dependency with target
2. **Random Forest Importance** - Tree-based feature importance
3. **Spearman Correlation** - Rank-based correlation with target
4. **Statistical Tests** - ANOVA F-test / F-regression

## Usage

### Command Line

```bash
# Basic: Select top 10 features
python transform_data.py train.csv --target label --select --select-top-k 10

# Full pipeline: Transform → Filter → Select
python transform_data.py train.csv \
  --target label \
  --filter \
  --select \
  --select-top-k 20 \
  --task classification
```

### Selection Criteria

**Option 1: Top K features**
```bash
--select-top-k 15  # Select top 15 features
```

**Option 2: Threshold-based**
```bash
--select-threshold 0.1  # Select features with score > 0.1
```

### Choose Methods

```bash
# All methods (ensemble - recommended)
--select-methods mutual_info tree_importance correlation statistical

# Specific methods only
--select-methods mutual_info correlation
```

### Task Type

```bash
--task classification  # Classification (default)
--task regression      # Regression
```

## Output Example

```
============================================================
FEATURE SELECTION
============================================================
Task: classification
Features: 12
Samples: 1000
Methods: mutual_info, tree_importance, correlation

Computing mutual_info scores...
  ✓ Computed mutual_info scores

Computing tree_importance scores...
  ✓ Computed tree_importance scores

Computing correlation scores...
  ✓ Computed correlation scores

============================================================
FEATURE SELECTION SUMMARY
============================================================

Total features evaluated: 12
Features selected: 5
Selection rate: 41.7%

============================================================
Top 20 Features (by aggregated score):
============================================================
   feature  aggregated_score  selected  mutual_info_score  tree_importance_score  correlation_score
       age            0.5244      True             0.5363                 0.1666             0.8704
age_binned_10         0.4980      True             0.4663                 0.1519             0.8757
age_capped            0.4742      True             0.4322                 0.1199             0.8704
    income            0.4515      True             0.3413                 0.1429             0.8704
income_capped         0.4419      True             0.2955                 0.1599             0.8704

✓ Feature selection complete
  Selected 5 out of 12 features
```

## Python API

```python
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig

# Configure
config = FeatureSelectionConfig(
    methods=['mutual_info', 'tree_importance', 'correlation'],
    top_k=15,
    task='classification'
)

# Fit and select
fs = FeatureSelector(config)
X_selected = fs.fit_transform(X, y)

# Get results
selected_features = fs.get_selected_features()
scores = fs.get_feature_scores()
top_10 = fs.get_top_features(k=10)

# Print summary
fs.print_summary(top_n=20)
```

## Integration with Pipeline

Feature selection works seamlessly with existing features:

```bash
python transform_data.py train.csv \
  --target label \                    # Exclude target from transforms
  --filter \                          # Remove low-quality features
  --select \                          # Rank & select best features
  --select-top-k 20                   # Keep top 20
```

**Pipeline flow:**
1. **Transform**: age → age_capped, age_binned_10, age_binned_20
2. **Filter**: Remove high missingness, high cardinality, zero variance
3. **Select**: Rank all features, select top 20
4. **Save**: Only selected features + target saved to output

## Files Added/Modified

### New Files
- `src/features/feature_selection.py` - Feature selection module
- `tests/test_feature_selection.py` - Test suite
- `FEATURE_SELECTION_GUIDE.md` - Complete usage guide
- `FEATURE_SELECTION_SUMMARY.md` - This file

### Modified Files
- `transform_data.py` - Added --select CLI options
- `README.md` - Updated with feature selection section
- `USAGE.md` - Added feature selection examples
- `QUICK_START.md` - Added feature selection quick start

## Testing

All tests pass:

```bash
python tests/test_feature_selection.py

# Tests:
✓ Mutual Information selection
✓ Random Forest importance selection
✓ Correlation-based selection
✓ Multi-method ensemble selection
✓ Transform (select columns)
✓ Feature scores retrieval
✓ Engineered features selection
```

## Benefits

✅ **Dimensionality Reduction** - Select only the most important features
✅ **Improved Performance** - Remove noise, keep signal
✅ **Faster Training** - Fewer features = faster models
✅ **Better Generalization** - Reduce overfitting
✅ **Interpretability** - Focus on features that matter
✅ **Ensemble Approach** - Multiple methods = robust selection

## Example Workflows

### Workflow 1: Quick Feature Ranking
```bash
# Fast correlation-based ranking
python transform_data.py train.csv --target label \
  --select --select-methods correlation --select-top-k 10
```

### Workflow 2: Comprehensive Selection
```bash
# Use all methods for robust selection
python transform_data.py train.csv --target label \
  --filter \
  --select \
  --select-methods mutual_info tree_importance correlation statistical \
  --select-top-k 20
```

### Workflow 3: Regression Task
```bash
python transform_data.py house_prices.csv --target price \
  --task regression \
  --select --select-top-k 15
```

## Next Steps

The feature selection completes the data preparation pipeline:

✅ Step 1: Feature Engineering (transform features)
✅ Step 2: Feature Filtering (remove low quality)
✅ Step 3: Feature Selection (select best for modeling)
⬜ Step 4: Model Training (TabNet, MLP, NN)
⬜ Step 5: Experiment Tracking (track features + models + metrics)

Ready to move to **modeling** (TabNet, MLP, NN with softmax)!
