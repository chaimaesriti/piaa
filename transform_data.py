#!/usr/bin/env python3
"""
Command-line tool to transform features from CSV data
Usage: python transform_data.py train.csv [options]
"""
import sys
import argparse
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig


def infer_column_types(df):
    """Automatically infer numerical and categorical columns"""
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical, categorical


def main():
    parser = argparse.ArgumentParser(description='Transform features in CSV data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--numerical', '-n', nargs='+', help='Numerical column names')
    parser.add_argument('--categorical', '-c', nargs='+', help='Categorical column names')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
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
    parser.add_argument('--max-cardinality-num', type=int, default=1000,
                        help='Max cardinality for numerical (default: 1000)')
    parser.add_argument('--max-cardinality-cat', type=int, default=100,
                        help='Max cardinality for categorical (default: 100)')
    parser.add_argument('--target', help='Target column to exclude from filtering')

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
            max_cardinality_numeric=args.max_cardinality_num,
            max_cardinality_categorical=args.max_cardinality_cat
        )

        ff = FeatureFilter(filter_config)

        # Get all transformed features
        transformed_numerical = fe.get_feature_list()
        # Separate back into numerical and categorical
        transformed_categorical = [col for col in df_transformed.columns
                                   if col in categorical_cols or col.endswith('_grouped')]

        print(f"\nApplying quality filters...")
        print(f"  Max missing rate: {args.max_missing:.0%}")
        print(f"  Max cardinality (numerical): {args.max_cardinality_num}")
        print(f"  Max cardinality (categorical): {args.max_cardinality_cat}")

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
    if args.output:
        print(f"\n{'='*60}")
        print(f"Saving to {args.output}...")
        df_transformed.to_csv(args.output, index=False)
        print(f"✓ Saved {len(df_transformed)} rows")
    else:
        print(f"\n{'='*60}")
        print("TIP: Use --output <file> to save transformed data")

    print(f"\n{'='*60}")
    print("DONE!")
    print('='*60)


if __name__ == "__main__":
    main()
