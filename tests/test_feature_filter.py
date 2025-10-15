"""
Test Feature Filter Module
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig


def create_problematic_data(n_samples=1000):
    """Create dataset with problematic features"""
    np.random.seed(42)

    # Good features
    good_age = np.random.randint(18, 80, n_samples)
    good_income = np.random.exponential(50000, n_samples)
    good_category = np.random.choice(['A', 'B', 'C', 'D'], n_samples)

    # High missingness feature (95% missing)
    high_missing = np.random.randn(n_samples)
    missing_mask = np.random.random(n_samples) > 0.05  # 95% missing
    high_missing[missing_mask] = np.nan

    # Medium missingness feature (50% missing)
    medium_missing = np.random.randn(n_samples)
    missing_mask = np.random.random(n_samples) > 0.50
    medium_missing[missing_mask] = np.nan

    # High cardinality numerical (unique IDs)
    high_card_numeric = np.arange(n_samples)

    # High cardinality categorical (200 unique categories)
    high_card_categorical = np.random.choice([f"cat_{i}" for i in range(200)], n_samples)

    # Zero variance feature
    zero_variance = np.ones(n_samples) * 5.0

    # Near-zero variance feature
    near_zero_variance = np.random.choice([1.0, 1.0001], n_samples)

    data = {
        # Good features
        'age': good_age,
        'income': good_income,
        'category': good_category,

        # Problematic features
        'high_missing_95': high_missing,
        'medium_missing_50': medium_missing,
        'user_id': high_card_numeric,
        'high_card_cat': high_card_categorical,
        'constant_value': zero_variance,
        'almost_constant': near_zero_variance,

        # Target
        'target': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)


def test_high_missingness_filter():
    """Test filtering features with high missingness"""
    print("=" * 60)
    print("Test 1: High Missingness Filter")
    print("=" * 60)

    df = create_problematic_data()

    print("\nMissing rates:")
    for col in df.columns:
        missing_rate = df[col].isna().sum() / len(df)
        if missing_rate > 0:
            print(f"  {col}: {missing_rate:.1%}")

    # Filter with 90% threshold
    config = FeatureFilterConfig(max_missing_rate=0.90)
    ff = FeatureFilter(config)
    ff.fit(df, target_col='target')

    print(f"\nThreshold: {config.max_missing_rate:.0%}")
    print(f"Features removed (high missingness): {ff.removed_features['high_missingness']}")
    print(f"Features kept: {len(ff.kept_features)}")

    # Verify
    assert 'high_missing_95' in ff.removed_features['high_missingness']
    assert 'medium_missing_50' not in ff.removed_features['high_missingness']

    print("\n✓ Test passed!")
    return ff


def test_high_cardinality_filter():
    """Test filtering features with high cardinality"""
    print("\n\n" + "=" * 60)
    print("Test 2: High Cardinality Filter")
    print("=" * 60)

    df = create_problematic_data()

    n_rows = len(df)
    print(f"\nTotal rows: {n_rows}")
    print("\nCardinality and ratios:")
    for col in df.columns:
        cardinality = df[col].nunique()
        ratio = cardinality / n_rows
        print(f"  {col}: {cardinality} unique ({ratio:.1%})")

    # Filter with ratio threshold (10% = remove features with >10% unique)
    config = FeatureFilterConfig(
        max_cardinality_ratio=0.10  # Remove if >10% unique values
    )
    ff = FeatureFilter(config)
    ff.fit(df, target_col='target')

    print(f"\nThreshold: {config.max_cardinality_ratio:.0%} (unique/rows)")
    print(f"Features removed (high cardinality): {ff.removed_features['high_cardinality']}")

    # Verify
    assert 'user_id' in ff.removed_features['high_cardinality']  # 100% unique
    assert 'high_card_cat' in ff.removed_features['high_cardinality']  # ~20% unique

    print("\n✓ Test passed!")
    return ff


def test_zero_variance_filter():
    """Test filtering features with zero variance"""
    print("\n\n" + "=" * 60)
    print("Test 3: Zero Variance Filter")
    print("=" * 60)

    df = create_problematic_data()

    print("\nVariance:")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col != 'target':
            variance = df[col].var()
            print(f"  {col}: {variance:.6f}")

    # Filter
    ff = FeatureFilter()
    ff.fit(df, target_col='target')

    print(f"\nFeatures removed (zero variance): {ff.removed_features['zero_variance']}")

    # Verify
    assert 'constant_value' in ff.removed_features['zero_variance']

    print("\n✓ Test passed!")
    return ff


def test_full_filter():
    """Test complete filtering pipeline"""
    print("\n\n" + "=" * 60)
    print("Test 4: Full Filter Pipeline")
    print("=" * 60)

    df = create_problematic_data()

    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original features: {list(df.columns)}")

    # Configure filter
    config = FeatureFilterConfig(
        max_missing_rate=0.90,
        max_cardinality_ratio=0.50,  # Remove if >50% unique
        min_variance=0.0
    )

    # Apply filter
    ff = FeatureFilter(config)
    df_filtered = ff.fit_transform(df, target_col='target')

    print(f"\nFiltered data shape: {df_filtered.shape}")
    print(f"Filtered features: {list(df_filtered.columns)}")

    # Print summary
    print()
    ff.print_summary()

    # Show detailed report
    print("\n" + "=" * 60)
    print("Detailed Filter Report:")
    print("=" * 60)
    summary = ff.get_filter_summary()
    summary_display = summary.sort_values('kept', ascending=True)
    print(summary_display.to_string(index=False))

    # Verify target is kept
    assert 'target' in df_filtered.columns
    assert 'age' in df_filtered.columns  # Good feature kept
    assert 'high_missing_95' not in df_filtered.columns  # Bad feature removed

    print("\n✓ Test passed!")
    return ff, df_filtered


def test_transform_consistency():
    """Test that transform works consistently on new data"""
    print("\n\n" + "=" * 60)
    print("Test 5: Transform Consistency")
    print("=" * 60)

    # Create train and test data
    df_train = create_problematic_data(1000)
    df_test = create_problematic_data(200)

    print(f"\nTrain data: {df_train.shape}")
    print(f"Test data: {df_test.shape}")

    # Fit on train
    ff = FeatureFilter()
    df_train_filtered = ff.fit_transform(df_train, target_col='target')

    print(f"\nTrain filtered: {df_train_filtered.shape}")
    print(f"Features kept: {ff.kept_features}")

    # Transform test with same filter
    df_test_filtered = ff.transform(df_test)

    print(f"Test filtered: {df_test_filtered.shape}")

    # Verify same columns
    assert list(df_train_filtered.columns) == list(df_test_filtered.columns)
    print("\n✓ Train and test have same features!")

    print("\n✓ Test passed!")
    return ff


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FEATURE FILTER TEST SUITE")
    print("=" * 60)

    # Run tests
    test_high_missingness_filter()
    test_high_cardinality_filter()
    test_zero_variance_filter()
    ff, df_filtered = test_full_filter()
    test_transform_consistency()

    print("\n\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
