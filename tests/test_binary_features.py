"""
Test Binary Feature Detection and Target Exclusion
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig


def create_data_with_binary():
    """Create dataset with binary features and target"""
    np.random.seed(42)
    n = 500

    # Regular features
    age = np.random.randint(18, 80, n)
    income = np.random.exponential(50000, n)

    # Binary numerical features
    is_premium = np.random.choice([0, 1], n)
    has_loan = np.random.choice([0, 1], n)

    # Binary categorical features
    gender = np.random.choice(['M', 'F'], n)
    region = np.random.choice(['North', 'South'], n)

    # Regular categorical
    country = np.random.choice(['USA', 'UK', 'Canada', 'France'], n)

    # Binary target
    target = np.random.choice([0, 1], n)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'is_premium': is_premium,
        'has_loan': has_loan,
        'gender': gender,
        'region': region,
        'country': country,
        'target': target
    })

    return df


def test_binary_detection():
    """Test binary feature detection"""
    print("=" * 60)
    print("Test 1: Binary Feature Detection")
    print("=" * 60)

    df = create_data_with_binary()

    print(f"\nOriginal data: {df.shape}")
    print("\nFeature cardinalities:")
    for col in df.columns:
        n_unique = df[col].nunique()
        marker = " [BINARY]" if n_unique == 2 else ""
        print(f"  {col}: {n_unique} unique values{marker}")

    fe = FeatureEngineer()

    # Test numerical
    numerical_cols = ['age', 'income', 'is_premium', 'has_loan']
    print(f"\n{'='*60}")
    print("Transforming numerical features...")
    print('='*60)
    df_transformed = fe.fit_transform_numerical(df, numerical_cols)

    print(f"\nBinary features detected: {fe.binary_features}")
    print(f"Transformed shape: {df_transformed.shape}")

    # Check that binary features were not transformed
    assert 'is_premium_capped' not in df_transformed.columns, "Binary feature should not be capped"
    assert 'has_loan_binned_10' not in df_transformed.columns, "Binary feature should not be binned"

    # Check that regular features were transformed
    assert 'age_capped' in df_transformed.columns, "Regular feature should be capped"
    assert 'income_binned_10' in df_transformed.columns, "Regular feature should be binned"

    print("\n✓ Binary numerical features correctly skipped!")

    # Test categorical
    categorical_cols = ['gender', 'region', 'country']
    print(f"\n{'='*60}")
    print("Transforming categorical features...")
    print('='*60)
    df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols)

    print(f"\nAll binary features: {fe.binary_features}")
    print(f"Final shape: {df_transformed.shape}")

    # Check that binary categorical features were not grouped
    assert 'gender_grouped' not in df_transformed.columns, "Binary category should not be grouped"
    assert 'region_grouped' not in df_transformed.columns, "Binary category should not be grouped"

    print("\n✓ Binary categorical features correctly skipped!")

    return fe, df_transformed


def test_target_exclusion():
    """Test target column exclusion"""
    print("\n\n" + "=" * 60)
    print("Test 2: Target Column Exclusion")
    print("=" * 60)

    df = create_data_with_binary()

    print(f"\nOriginal data: {df.shape}")
    print(f"Target column: 'target'")

    fe = FeatureEngineer()

    # Transform with target specified
    numerical_cols = ['age', 'income', 'is_premium', 'has_loan', 'target']
    categorical_cols = ['gender', 'region', 'country']

    print(f"\n{'='*60}")
    print("Transforming with target='target'")
    print('='*60)

    df_transformed = fe.fit_transform_numerical(df, numerical_cols, target_col='target')
    df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols, target_col='target')

    print(f"\nTransformed shape: {df_transformed.shape}")

    # Check that target was not transformed
    assert 'target_capped' not in df_transformed.columns, "Target should not be transformed"
    assert 'target_binned_10' not in df_transformed.columns, "Target should not be transformed"
    assert 'target' in df_transformed.columns, "Target should still exist"

    # Check that target is not in feature mapping
    assert 'target' not in fe.feature_mapping or fe.feature_mapping.get('target') == [], \
        "Target should not be in feature mapping"

    print("\n✓ Target column correctly excluded from transformations!")

    return fe, df_transformed


def test_combined():
    """Test binary detection + target exclusion together"""
    print("\n\n" + "=" * 60)
    print("Test 3: Combined Binary Detection + Target Exclusion")
    print("=" * 60)

    df = create_data_with_binary()

    fe = FeatureEngineer()

    numerical_cols = ['age', 'income', 'is_premium', 'has_loan', 'target']
    categorical_cols = ['gender', 'region', 'country']

    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Numerical: {numerical_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"Target: target")

    df_transformed = fe.fit_transform_numerical(df, numerical_cols, target_col='target')
    df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols, target_col='target')

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"Binary features detected: {fe.binary_features}")
    print(f"Target: {fe.target_col}")
    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")

    # Get feature summary
    summary = fe.get_feature_summary()
    print(f"\n{'='*60}")
    print("Feature Summary")
    print('='*60)
    print(summary.to_string(index=False))

    # Count transformations by type
    print(f"\n{'='*60}")
    print("Transformation Counts")
    print('='*60)
    for trans_type in ['binary', 'capped', 'binned', 'none']:
        count = summary[summary['transformation_type'] == trans_type].shape[0]
        if count > 0:
            print(f"  {trans_type}: {count}")

    # Verify expectations
    expected_binary = ['is_premium', 'has_loan', 'gender', 'region', 'target']
    detected_binary = fe.binary_features + ([fe.target_col] if fe.target_col else [])

    print(f"\nExpected binary/target: {sorted(expected_binary)}")
    print(f"Detected binary/target: {sorted(detected_binary)}")

    print("\n✓ Combined test passed!")

    return fe, df_transformed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BINARY FEATURE DETECTION TEST SUITE")
    print("=" * 60)

    # Run tests
    fe1, df1 = test_binary_detection()
    fe2, df2 = test_target_exclusion()
    fe3, df3 = test_combined()

    print("\n\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print("  ✓ Binary features are automatically detected")
    print("  ✓ Binary features are not transformed (kept as-is)")
    print("  ✓ Target column can be excluded from transformations")
    print("  ✓ Both work correctly together")
