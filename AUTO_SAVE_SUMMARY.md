# Auto-Save Feature Summary

## What Was Added

**Automatic file saving** with smart naming convention for transformed data.

## Default Behavior

When you run:
```bash
python transform_data.py train.csv
```

The transformed data is **automatically saved** as:
```
transformed_train.csv
```

## Naming Convention

| Input File | Auto-Saved As |
|------------|---------------|
| `train.csv` | `transformed_train.csv` |
| `data/train.csv` | `data/transformed_train.csv` |
| `/path/to/file.csv` | `/path/to/transformed_file.csv` |
| `my_dataset.csv` | `transformed_my_dataset.csv` |

The file is saved in the **same directory** as the input file with `transformed_` prefix.

## Control Options

### 1. Custom Filename (Override Default)
```bash
python transform_data.py train.csv --output my_custom_name.csv
```
Saves to `my_custom_name.csv` instead of `transformed_train.csv`

### 2. Disable Auto-Save (Preview Only)
```bash
python transform_data.py train.csv --no-save
```
Shows transformations but doesn't save any file

## Examples

### Basic Usage
```bash
# Input: data/train.csv
python transform_data.py data/train.csv --target label

# Output automatically saved to: data/transformed_train.csv
```

### With Filtering
```bash
# Input: train.csv
python transform_data.py train.csv --target label --filter

# Output automatically saved to: transformed_train.csv
```

### Custom Output Location
```bash
# Input: train.csv
python transform_data.py train.csv --output results/clean_data.csv

# Output saved to: results/clean_data.csv (custom location)
```

### Preview Only
```bash
# Input: train.csv
python transform_data.py train.csv --no-save

# No file saved, just displays transformations
```

## Output Message

When auto-saving, you'll see:

```
============================================================
SAVING TRANSFORMED DATA
============================================================
Output file: data/transformed_train.csv
✓ Saved 1000 rows, 25 columns
✓ File: data/transformed_train.csv
```

When using `--no-save`:

```
============================================================
AUTO-SAVE DISABLED
============================================================
ℹ  Data not saved (--no-save specified)
   Use --output <file> to save manually
```

## Benefits

✅ **No manual output specification needed** - just run and go
✅ **Consistent naming** - easy to identify transformed files
✅ **Same directory** - output stays with input
✅ **Still flexible** - can override with `--output` or disable with `--no-save`
✅ **Clear feedback** - always shows where file was saved

## Code Changes

### Files Modified
- `transform_data.py` - Added auto-save logic and `--no-save` flag

### New Functions
- `generate_output_filename(input_path)` - Generates output filename with `transformed_` prefix

### New Arguments
- `--no-save` - Disable automatic saving
- Updated `--output` help text to mention auto-save default

## Testing

Verified with:
- ✅ Basic auto-save: `train.csv` → `transformed_train.csv`
- ✅ Subdirectory: `data/train.csv` → `data/transformed_train.csv`
- ✅ Custom output: `--output custom.csv` works
- ✅ No-save: `--no-save` doesn't create file
- ✅ Integration: Works with all other features (binary detection, filtering, etc.)

See `tests/test_autosave.py` for test suite.

## Documentation Updated

- ✅ `README.md` - Added auto-save examples
- ✅ `USAGE.md` - Documented auto-save behavior and options
- ✅ `QUICK_START.md` - Updated all examples to show auto-save
- ✅ `CHANGELOG.md` - Documented new feature
- ✅ `AUTO_SAVE_SUMMARY.md` - This file

## Backwards Compatibility

✅ **Fully backwards compatible**
- Old behavior: Required `--output` to save → Now auto-saves by default
- Can still use `--output` for custom filenames
- Add `--no-save` to get old preview-only behavior
