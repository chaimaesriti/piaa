#!/usr/bin/env python3
"""
Demo: Binary Feature Detection and Target Exclusion
Run this to see how binary features and target columns are handled
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_engineering import FeatureEngineer


def main():
    print("=" * 70)
    print("DEMO: Binary Feature Detection & Target Exclusion")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        # Regular numerical (will be transformed)
        'age': np.random.randint(18, 80, n),
        'salary': np.random.exponential(50000, n),

        # Binary numerical (will NOT be transformed)
        'is_employed': np.random.choice([0, 1], n),
        'has_degree': np.random.choice([0, 1], n),

        # Regular categorical (will be transformed if rare categories)
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),

        # Binary categorical (will NOT be transformed)
        'gender': np.random.choice(['M', 'F'], n),
        'premium': np.random.choice(['Yes', 'No'], n),

        # Target (binary, will be EXCLUDED from transformations)
        'target': np.random.choice([0, 1], n)
    })

    print(f"\nSample data created: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n" + "=" * 70)
    print("Feature Cardinalities:")
    print("=" * 70)
    for col in df.columns:
        n_unique = df[col].nunique()
        marker = " ← BINARY" if n_unique == 2 else ""
        target_marker = " ← TARGET" if col == 'target' else ""
        print(f"  {col:15s}: {n_unique:3d} unique values{marker}{target_marker}")

    # Transform features
    print("\n" + "=" * 70)
    print("Applying Transformations with target='target'")
    print("=" * 70)

    fe = FeatureEngineer()

    numerical_cols = ['age', 'salary', 'is_employed', 'has_degree', 'target']
    categorical_cols = ['city', 'gender', 'premium']

    df_transformed = fe.fit_transform_numerical(df, numerical_cols, target_col='target')
    df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols, target_col='target')

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Original shape:    {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"Features added:    {df_transformed.shape[1] - df.shape[1]}")

    print("\n" + "=" * 70)
    print("What was skipped:")
    print("=" * 70)
    print(f"Binary features: {fe.binary_features}")
    print(f"Target column:   {fe.target_col}")

    print("\n" + "=" * 70)
    print("What was transformed:")
    print("=" * 70)
    summary = fe.get_feature_summary()

    # Show by type
    for feat_type in ['binary', 'capped', 'binned', 'categorical_grouped', 'none']:
        subset = summary[summary['transformation_type'] == feat_type]
        if not subset.empty:
            print(f"\n{feat_type.upper()}:")
            for _, row in subset.iterrows():
                if row['original_feature'] != row['transformed_feature']:
                    print(f"  {row['original_feature']} → {row['transformed_feature']}")
                else:
                    print(f"  {row['original_feature']} (unchanged)")

    print("\n" + "=" * 70)
    print("New Column Names:")
    print("=" * 70)
    new_cols = [col for col in df_transformed.columns if col not in df.columns]
    print(f"\nAdded {len(new_cols)} new columns:")
    for col in new_cols:
        print(f"  - {col}")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("  ✓ Binary features (2 unique values) are auto-detected")
    print("  ✓ Binary features are kept as-is, no transformations applied")
    print("  ✓ Target column is excluded when specified")
    print("  ✓ Regular features get capped + binned variants")
    print("  ✓ All transformations are tracked for reproducibility")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
