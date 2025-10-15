# Feature Engineering CLI Usage

## Auto-Save Behavior

**By default, transformed data is automatically saved** with the naming convention:
- Input: `train.csv` → Output: `transformed_train.csv`
- Input: `data/train.csv` → Output: `data/transformed_train.csv`

**Control auto-save:**
- Use `--output <file>` to specify custom filename
- Use `--no-save` to disable saving (preview only)

## Quick Start

### 1. Auto-detect columns (easiest)
```bash
python transform_data.py train.csv
```
**Automatically saves to:** `transformed_train.csv`

### 2. With target column (recommended)
```bash
python transform_data.py data/train.csv --target label
```
**Automatically saves to:** `data/transformed_train.csv`

Automatically:
- Excludes target from transformations
- Detects binary features (2 unique values) and keeps them as-is
- Shows what was skipped
- Saves with `transformed_` prefix in same directory

### 3. With feature quality filtering
```bash
python transform_data.py data/train.csv --filter --target label
```
Removes:
- Features with >90% missing values
- Features with >90% unique values / total rows (adapts to dataset size)
- Zero variance features

### 3a. With feature selection (rank & select best)
```bash
python transform_data.py data/train.csv --target label --select --select-top-k 10
```
Selects top 10 features using:
- Mutual Information
- Random Forest importance
- Correlation with target

### 3b. Full pipeline (transform → filter → select)
```bash
python transform_data.py data/train.csv --target label --filter --select --select-top-k 20
```
Complete pipeline: engineer features → filter quality → select best for modeling

### 4. Specify columns manually
```bash
python transform_data.py data/train.csv \
  --numerical age income credit_score \
  --categorical country product
```

### 5. Custom output filename
```bash
python transform_data.py train.csv --output my_custom_output.csv
```
Override the default `transformed_train.csv` filename

### 6. Disable auto-save
```bash
python transform_data.py train.csv --no-save
```
Preview transformations without saving any file

### 7. Show detailed summary
```bash
python transform_data.py data/train.csv --show-summary
```

## Advanced Options

### Custom binning
```bash
python transform_data.py data/train.csv --bins 5 10 20 50
```
Creates: `feature_binned_5`, `feature_binned_10`, `feature_binned_20`, `feature_binned_50`

### Custom capping percentiles
```bash
python transform_data.py data/train.csv --cap-percentiles 5 95
```
Caps at 5th and 95th percentiles (less aggressive than default 1/99)

### Custom category frequency threshold
```bash
python transform_data.py data/train.csv --min-freq 0.05
```
Groups categories with <5% frequency (default is 1%)

### Custom feature filtering thresholds
```bash
python transform_data.py data/train.csv --filter \
  --max-missing 0.80 \
  --max-cardinality-ratio 0.75
```
- `--max-missing`: Max missing rate (default: 0.90 = 90%)
- `--max-cardinality-ratio`: Max ratio of unique/rows (default: 0.90 = 90%)

**Examples:**
- `--max-cardinality-ratio 0.50`: Remove features with >50% unique values
- `--max-cardinality-ratio 0.95`: Only remove near-ID features (>95% unique)
- Automatically adapts to dataset size

### Filtering with target column
```bash
python transform_data.py data/train.csv --filter --target label
```
The target column will be excluded from filtering checks

### Feature selection options
```bash
# Select top K features
python transform_data.py data/train.csv --target label --select --select-top-k 15

# Select by threshold
python transform_data.py data/train.csv --target label --select --select-threshold 0.1

# Choose selection methods
python transform_data.py data/train.csv --target label --select \
  --select-methods mutual_info tree_importance correlation statistical \
  --select-top-k 10

# For regression tasks
python transform_data.py data/train.csv --target price --select \
  --task regression \
  --select-top-k 15
```

Available selection methods:
- `mutual_info`: Mutual information score
- `tree_importance`: Random Forest feature importance
- `correlation`: Spearman correlation with target
- `statistical`: ANOVA F-test (classification) or F-regression (regression)

### All together
```bash
python transform_data.py data/train.csv \
  --numerical age income \
  --categorical country \
  --bins 10 20 \
  --cap-percentiles 1 99 \
  --min-freq 0.01 \
  --output data/train_transformed.csv \
  --show-summary
```

## Help
```bash
python transform_data.py --help
```
