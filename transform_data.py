#!/usr/bin/env python3
"""
Command-line tool to transform features from CSV data
Usage: python transform_data.py train.csv [options]
"""
import sys
import os
import argparse
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig


def infer_column_types(df):
    """Automatically infer numerical and categorical columns"""
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical, categorical


def generate_output_filename(input_path):
    """
    Generate output filename with transformed_ prefix

    Examples:
        train.csv -> transformed_train.csv
        data/train.csv -> data/transformed_train.csv
        /path/to/file.csv -> /path/to/transformed_file.csv
    """
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    output_filename = f"transformed_{name}{ext}"

    if directory:
        return os.path.join(directory, output_filename)
    else:
        return output_filename


def main():
    parser = argparse.ArgumentParser(description='Transform features in CSV data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--numerical', '-n', nargs='+', help='Numerical column names')
    parser.add_argument('--categorical', '-c', nargs='+', help='Categorical column names')
    parser.add_argument('--output', '-o', help='Output file path (default: auto-generated as transformed_{input_name}.csv)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not automatically save transformed data')
    parser.add_argument('--cap-percentiles', nargs=2, type=float, default=[1, 99],
                        help='Percentiles for capping (default: 1 99)')
    parser.add_argument('--bins', nargs='+', type=int, default=[10, 20],
                        help='Number of bins (default: 10 20)')
    parser.add_argument('--min-freq', type=float, default=0.01,
                        help='Minimum category frequency (default: 0.01)')
    parser.add_argument('--show-summary', action='store_true',
                        help='Show transformation summary')

    # Feature filtering options
    parser.add_argument('--filter', action='store_true',
                        help='Enable feature quality filtering')
    parser.add_argument('--max-missing', type=float, default=0.90,
                        help='Max missing rate (default: 0.90)')
    parser.add_argument('--max-cardinality-ratio', type=float, default=0.90,
                        help='Max cardinality ratio: nunique/nrows (default: 0.90 = 90%%)')
    parser.add_argument('--target', help='Target column to exclude from filtering')

    # Feature selection options
    parser.add_argument('--select', action='store_true',
                        help='Enable feature selection to rank best features')
    parser.add_argument('--select-methods', nargs='+',
                        default=['mutual_info', 'tree_importance', 'correlation'],
                        help='Feature selection methods (default: mutual_info tree_importance correlation)')
    parser.add_argument('--select-top-k', type=int,
                        help='Select top K features')
    parser.add_argument('--select-threshold', type=float,
                        help='Select features with score above threshold')
    parser.add_argument('--select-max-corr', type=float, default=0.95,
                        help='Max correlation between selected features (default: 0.95)')
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification',
                        help='ML task type (default: classification)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)

    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nData types:")
    print(df.dtypes)

    # Determine column types
    if args.numerical or args.categorical:
        numerical_cols = args.numerical or []
        categorical_cols = args.categorical or []
        print(f"\nUsing specified columns:")
    else:
        numerical_cols, categorical_cols = infer_column_types(df)
        print(f"\nAuto-detected columns:")

    print(f"  Numerical ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical ({len(categorical_cols)}): {categorical_cols}")

    # Create config
    config = FeatureTransformConfig(
        cap_percentiles=tuple(args.cap_percentiles),
        n_bins_options=args.bins,
        min_category_freq=args.min_freq
    )

    # Initialize feature engineer
    fe = FeatureEngineer(config)

    # Transform features
    print(f"\n{'='*60}")
    print("TRANSFORMING FEATURES")
    print('='*60)

    df_transformed = df.copy()

    # Numerical transformations
    if numerical_cols:
        print(f"\nTransforming {len(numerical_cols)} numerical features...")
        df_transformed = fe.fit_transform_numerical(df_transformed, numerical_cols, target_col=args.target)
        print(f"✓ Created numerical transformations")

    # Categorical transformations
    if categorical_cols:
        print(f"\nTransforming {len(categorical_cols)} categorical features...")
        df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols, target_col=args.target)
        print(f"✓ Processed categorical features")

    # Feature filtering (optional)
    if args.filter:
        print(f"\n{'='*60}")
        print("FILTERING FEATURES")
        print('='*60)

        filter_config = FeatureFilterConfig(
            max_missing_rate=args.max_missing,
            max_cardinality_ratio=args.max_cardinality_ratio
        )

        ff = FeatureFilter(filter_config)

        print(f"\nApplying quality filters...")
        print(f"  Max missing rate: {args.max_missing:.0%}")
        print(f"  Max cardinality ratio: {args.max_cardinality_ratio:.0%} (unique/rows)")

        df_transformed = ff.fit_transform(
            df_transformed,
            numerical_cols=None,  # Auto-detect
            categorical_cols=None,  # Auto-detect
            target_col=args.target
        )

        print(f"\n✓ Filtering complete")
        ff.print_summary()
    else:
        print(f"\nℹ  Filtering disabled. Use --filter to enable quality filtering.")

    # Feature selection (optional)
    if args.select:
        if args.target is None:
            print(f"\n✗ Error: --target is required for feature selection")
            sys.exit(1)

        if args.target not in df_transformed.columns:
            print(f"\n✗ Error: Target column '{args.target}' not found in data")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("FEATURE SELECTION")
        print('='*60)

        # Separate features and target
        X = df_transformed.drop(columns=[args.target])
        y = df_transformed[args.target]

        # Configure feature selector
        selection_config = FeatureSelectionConfig(
            methods=args.select_methods,
            top_k=args.select_top_k,
            threshold=args.select_threshold,
            max_correlation=args.select_max_corr,
            task=args.task
        )

        fs = FeatureSelector(selection_config)
        fs.fit(X, y)

        # Print summary
        fs.print_summary(top_n=20)

        # Select features
        X_selected = fs.transform(X)

        # Reconstruct dataframe with selected features + target
        df_transformed = pd.concat([X_selected, y], axis=1)

        print(f"\n✓ Feature selection complete")
        print(f"  Selected {len(X_selected.columns)} out of {len(X.columns)} features")

    else:
        print(f"\nℹ  Feature selection disabled. Use --select to enable feature selection.")

    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Original shape:    {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"New features:      {df_transformed.shape[1] - df.shape[1]}")

    # Show feature list
    print(f"\n{'='*60}")
    print("TRANSFORMED FEATURES")
    print('='*60)
    all_features = fe.get_feature_list()
    for i, feat in enumerate(all_features, 1):
        print(f"{i:3d}. {feat}")

    # Show transformation summary
    if args.show_summary:
        print(f"\n{'='*60}")
        print("TRANSFORMATION SUMMARY")
        print('='*60)
        summary = fe.get_feature_summary()
        print(summary.to_string(index=False))

    # Show sample of transformed data
    print(f"\n{'='*60}")
    print("SAMPLE OF TRANSFORMED DATA (first 5 rows)")
    print('='*60)
    print(df_transformed.head())

    # Save output
    print(f"\n{'='*60}")
    if args.no_save:
        print("AUTO-SAVE DISABLED")
        print('='*60)
        print("ℹ  Data not saved (--no-save specified)")
        print("   Use --output <file> to save manually")
    else:
        # Determine output filename
        if args.output:
            output_file = args.output
        else:
            output_file = generate_output_filename(args.input_file)

        print(f"SAVING TRANSFORMED DATA")
        print('='*60)
        print(f"Output file: {output_file}")

        try:
            df_transformed.to_csv(output_file, index=False)
            print(f"✓ Saved {len(df_transformed)} rows, {len(df_transformed.columns)} columns")
            print(f"✓ File: {output_file}")
        except Exception as e:
            print(f"✗ Error saving file: {e}")

    print(f"\n{'='*60}")
    print("DONE!")
    print('='*60)


if __name__ == "__main__":
    main()
