# Cardinality Filtering Update - Ratio-Based Approach

## What Changed

**Cardinality filtering now uses a ratio-based threshold** instead of fixed thresholds.

### Before (Fixed Thresholds)
```python
# Old approach
max_cardinality_numeric = 1000    # Remove if >1000 unique values
max_cardinality_categorical = 100 # Remove if >100 unique categories
```

**Problem**: Not adaptive to dataset size
- 1000 unique values in 1000 rows (100% unique) = ID column → Should remove
- 1000 unique values in 1,000,000 rows (0.1% unique) = Normal feature → Should keep

### After (Ratio-Based)
```python
# New approach
max_cardinality_ratio = 0.90  # Remove if nunique/nrows > 90%
```

**Benefit**: Adapts automatically to dataset size
- 900 unique in 1000 rows (90% ratio) → Removed
- 1000 unique in 10,000 rows (10% ratio) → Kept
- 10,000 unique in 100,000 rows (10% ratio) → Kept

## Usage

### CLI

```bash
# Default: Remove features with >90% unique values
python transform_data.py train.csv --filter

# Custom ratio: Remove if >50% unique
python transform_data.py train.csv --filter --max-cardinality-ratio 0.50

# Stricter: Only remove near-ID features (>95% unique)
python transform_data.py train.csv --filter --max-cardinality-ratio 0.95
```

### Python API

```python
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig

# Configure with ratio
config = FeatureFilterConfig(
    max_missing_rate=0.90,
    max_cardinality_ratio=0.90  # Remove if >90% unique/rows
)

ff = FeatureFilter(config)
df_filtered = ff.fit_transform(df, target_col='label')
```

## Examples

### Example 1: 1000-row dataset
```
Feature          | Unique Values | Ratio  | Action (90% threshold)
-----------------|---------------|--------|----------------------
user_id          | 1000          | 100%   | ✗ REMOVE
transaction_id   | 950           | 95%    | ✗ REMOVE
email            | 920           | 92%    | ✗ REMOVE
age              | 62            | 6.2%   | ✓ KEEP
city             | 50            | 5%     | ✓ KEEP
country          | 10            | 1%     | ✓ KEEP
```

### Example 2: 100,000-row dataset
```
Feature          | Unique Values | Ratio  | Action (90% threshold)
-----------------|---------------|--------|----------------------
user_id          | 100,000       | 100%   | ✗ REMOVE
transaction_id   | 95,000        | 95%    | ✗ REMOVE
postal_code      | 5,000         | 5%     | ✓ KEEP
age              | 80            | 0.08%  | ✓ KEEP
```

## Output

The filter now shows the cardinality ratio:

```
============================================================
FEATURE FILTER SUMMARY
============================================================
Total features:            12
Kept features:             6 (50.0%)
Removed features:          6

Removal reasons:
  - High missingness (>90%): 2
  - High cardinality (>90% unique/rows): 3
  - Zero variance:           1

Removed features:

  high_cardinality:
    - user_id (cardinality: 1000, ratio: 100.0%)
    - income (cardinality: 950, ratio: 95.0%)
    - zip_code (cardinality: 920, ratio: 92.0%)
```

## Migration Guide

### CLI Users

**Before:**
```bash
python transform_data.py train.csv --filter \
  --max-cardinality-num 500 \
  --max-cardinality-cat 50
```

**After:**
```bash
python transform_data.py train.csv --filter \
  --max-cardinality-ratio 0.90
```

### Python API Users

**Before:**
```python
config = FeatureFilterConfig(
    max_cardinality_numeric=1000,
    max_cardinality_categorical=100
)
```

**After:**
```python
config = FeatureFilterConfig(
    max_cardinality_ratio=0.90
)
```

## Recommended Values

| Threshold | Use Case | Description |
|-----------|----------|-------------|
| 0.50 | Conservative | Remove if >50% unique (strict filtering) |
| 0.75 | Moderate | Remove if >75% unique (balanced) |
| **0.90** | **Default** | **Remove if >90% unique (recommended)** |
| 0.95 | Lenient | Only remove near-ID features |
| 0.99 | Very Lenient | Only remove actual IDs |

## Benefits

✅ **Dataset Size Agnostic**: Works with any dataset size
✅ **More Intuitive**: Ratio easier to understand than absolute numbers
✅ **Better Default**: 90% threshold works well for most datasets
✅ **Prevents Errors**: Won't accidentally remove valid features in large datasets
✅ **Simpler API**: One parameter instead of two

## Testing

All tests updated and passing:
```bash
python tests/test_feature_filter.py

✓ High missingness filter
✓ High cardinality filter (ratio-based)
✓ Zero variance filter
✓ Full pipeline
✓ Transform consistency
```

## Files Modified

- `src/features/feature_filter.py` - Updated to use ratio-based filtering
- `transform_data.py` - Updated CLI argument from `--max-cardinality-num/cat` to `--max-cardinality-ratio`
- `tests/test_feature_filter.py` - Updated tests to use ratio
- `README.md` - Updated documentation
- `USAGE.md` - Updated usage examples
- `QUICK_START.md` - Updated quick start guide
- `CARDINALITY_RATIO_UPDATE.md` - This document

## Backwards Compatibility

⚠️ **Breaking Change**: Old parameters `--max-cardinality-num` and `--max-cardinality-cat` no longer exist.

**Migration is simple**: Replace with `--max-cardinality-ratio 0.90`

Most users won't be affected as they were using default values.
