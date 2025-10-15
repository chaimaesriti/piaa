# Feature Engineering CLI Usage

## Quick Start

### 1. Auto-detect columns (easiest)
```bash
python transform_data.py data/train.csv
```

### 2. With target column (recommended)
```bash
python transform_data.py data/train.csv --target label
```
Automatically:
- Excludes target from transformations
- Detects binary features (2 unique values) and keeps them as-is
- Shows what was skipped

### 3. With feature quality filtering
```bash
python transform_data.py data/train.csv --filter --target label
```
Removes:
- Features with >90% missing values
- Numerical features with ≥1000 unique values
- Categorical features with ≥100 unique values
- Zero variance features

### 3. Specify columns manually
```bash
python transform_data.py data/train.csv \
  --numerical age income credit_score \
  --categorical country product
```

### 4. Save transformed data
```bash
python transform_data.py data/train.csv --output data/train_transformed.csv
```

### 5. Show detailed summary
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
  --max-cardinality-num 500 \
  --max-cardinality-cat 50
```
- `--max-missing`: Max missing rate (default: 0.90)
- `--max-cardinality-num`: Max unique values for numerical (default: 1000)
- `--max-cardinality-cat`: Max unique values for categorical (default: 100)

### Filtering with target column
```bash
python transform_data.py data/train.csv --filter --target label
```
The target column will be excluded from filtering checks

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
