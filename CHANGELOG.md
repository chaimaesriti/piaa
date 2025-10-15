# Changelog

## Auto-Save Feature Added

### New Feature: Automatic Save with Smart Naming

**Transformed data is now automatically saved** with the naming convention `transformed_{input_name}.csv`

#### Examples:
- Input: `train.csv` → Auto-saves to: `transformed_train.csv`
- Input: `data/train.csv` → Auto-saves to: `data/transformed_train.csv`
- Input: `/path/to/file.csv` → Auto-saves to: `/path/to/transformed_file.csv`

#### Usage:
```bash
# Auto-save (default behavior)
python transform_data.py train.csv

# Custom filename
python transform_data.py train.csv --output my_output.csv

# Preview only (no save)
python transform_data.py train.csv --no-save
```

#### Benefits:
- ✓ No need to specify `--output` every time
- ✓ Consistent naming convention
- ✓ Files saved in same directory as input
- ✓ Easy to identify transformed files
- ✓ Can still override with `--output` or disable with `--no-save`

---

## Binary Feature Detection & Target Exclusion Added

### New Features

#### 1. **Automatic Binary Feature Detection**
- Detects features with exactly 2 unique values
- Works for both numerical (0/1) and categorical (Yes/No, M/F, etc.)
- Binary features are **kept as-is** without transformation
- No capping, binning, or grouping applied

#### 2. **Target Column Exclusion**
- New parameter: `target_col` in `fit_transform_numerical()` and `fit_transform_categorical()`
- Target is automatically excluded from all transformations
- Target preserved in final dataset unchanged
- Prevents target leakage during feature engineering

#### 3. **Enhanced Logging**
- Shows which features are detected as binary
- Shows when target column is skipped
- Clear feedback about what's being transformed vs kept

### Usage

#### Command Line
```bash
# Specify target to exclude it from transformations
python transform_data.py train.csv --target label

# With filtering
python transform_data.py train.csv --target label --filter
```

#### Python API
```python
fe = FeatureEngineer()

# Pass target_col to exclude target from transformations
df_transformed = fe.fit_transform_numerical(
    df,
    numerical_cols,
    target_col='target'
)

df_transformed = fe.fit_transform_categorical(
    df_transformed,
    categorical_cols,
    target_col='target'
)

# Check what was detected
print(f"Binary features: {fe.binary_features}")
print(f"Target: {fe.target_col}")
```

### Examples

**Before:**
```
age (62 unique) → age_capped, age_binned_10, age_binned_20
is_premium (2 unique) → is_premium_capped, is_premium_binned_10, is_premium_binned_20
target (2 unique) → target_capped, target_binned_10, target_binned_20
```

**After:**
```
age (62 unique) → age_capped, age_binned_10, age_binned_20
is_premium (2 unique) → is_premium (unchanged, binary)
target (2 unique) → target (unchanged, excluded)
```

### Benefits

1. **Prevents unnecessary transformations** on binary features
2. **Protects target column** from accidental transformation
3. **Cleaner feature sets** with fewer redundant columns
4. **Better model interpretability** - binary features stay binary
5. **Automatic detection** - no manual specification needed

### Files Modified

- `src/features/feature_engineering.py` - Added binary detection and target exclusion
- `transform_data.py` - Updated to pass target_col parameter
- `README.md` - Documented new features
- `USAGE.md` - Added usage examples

### Files Added

- `tests/test_binary_features.py` - Test suite for binary detection
- `demo_binary_detection.py` - Interactive demo script
- `CHANGELOG.md` - This file

### Testing

```bash
# Run binary feature tests
python tests/test_binary_features.py

# Run interactive demo
python demo_binary_detection.py
```

All existing tests still pass.
