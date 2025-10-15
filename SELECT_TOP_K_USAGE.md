# How to Use --select-top-k

## The Error You Saw

```bash
zsh: command not found: --select-top-k
```

This means you tried to run `--select-top-k` as a standalone command. It needs to be part of the `python transform_data.py` command.

## ❌ WRONG (Don't do this)

```bash
python transform_data.py train.csv --target Response
--select-top-k 20  # ← This is on a new line, shell treats it as a separate command
```

## ✅ CORRECT Usage

### Option 1: All on one line
```bash
python transform_data.py train.csv --target Response --select --select-top-k 20
```

### Option 2: Multi-line with backslash
```bash
python transform_data.py train.csv \
  --target Response \
  --select \
  --select-top-k 20
```

**Note the backslash `\` at the end of each line!**

## For Your Dataset

You have **279 columns** after transformation. Here's how to select the best features:

### Select Top 20 Features
```bash
python transform_data.py train.csv \
  --target Response \
  --select \
  --select-top-k 20
```

### Select Top 30 Features
```bash
python transform_data.py train.csv \
  --target Response \
  --select \
  --select-top-k 30
```

### Select Top 50 Features
```bash
python transform_data.py train.csv \
  --target Response \
  --select \
  --select-top-k 50
```

### Full Pipeline: Filter THEN Select
```bash
python transform_data.py train.csv \
  --target Response \
  --filter \
  --select \
  --select-top-k 30
```

This will:
1. Transform features (done - you have 279 features)
2. Filter low-quality features (remove high cardinality, high missingness)
3. Select top 30 from remaining features
4. Save to `transformed_train.csv`

## What You'll Get

**Before selection:**
- 279 columns (278 features + 1 target)
- 59,381 rows

**After `--select-top-k 20`:**
- 21 columns (20 features + 1 target)
- 59,381 rows
- Only the most important features

## Check Your Current Data

```bash
# See what columns you have now
head -1 transformed_train.csv | tr ',' '\n' | wc -l
```

## Re-run with Selection

If you want to select top features from your existing `train.csv`:

```bash
# Select top 30 features
python transform_data.py train.csv --target Response --select --select-top-k 30 --output train_top30.csv

# Select top 50 features
python transform_data.py train.csv --target Response --select --select-top-k 50 --output train_top50.csv
```

## Common Mistakes

### 1. Missing backslash in multi-line
```bash
# ❌ WRONG - missing backslash
python transform_data.py train.csv
  --target Response
  --select-top-k 20
```

```bash
# ✅ CORRECT - backslash at end of each line
python transform_data.py train.csv \
  --target Response \
  --select-top-k 20
```

### 2. Forgetting --select flag
```bash
# ❌ WRONG - missing --select flag
python transform_data.py train.csv --target Response --select-top-k 20
```

```bash
# ✅ CORRECT - needs --select flag first
python transform_data.py train.csv --target Response --select --select-top-k 20
```

### 3. Running option as standalone command
```bash
# ❌ WRONG - this tries to run --select-top-k as a program
--select-top-k 20
```

```bash
# ✅ CORRECT - it's part of the python command
python transform_data.py train.csv --target Response --select --select-top-k 20
```

## Quick Reference

### Minimal (select from existing transformed data)
```bash
python transform_data.py train.csv --target Response --select --select-top-k 20
```

### With filtering
```bash
python transform_data.py train.csv --target Response --filter --select --select-top-k 20
```

### Custom methods
```bash
python transform_data.py train.csv --target Response --select --select-methods mutual_info correlation --select-top-k 20
```

### Different output file
```bash
python transform_data.py train.csv --target Response --select --select-top-k 20 --output train_selected.csv
```

## Verify It Works

Test with a simple command first:
```bash
python transform_data.py train.csv --target Response --select --select-top-k 5 --no-save
```

This will show you what would be selected without saving a file.
