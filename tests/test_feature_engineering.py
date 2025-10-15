"""
Test Feature Engineering Module
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig


def create_sample_data(n_samples=1000):
    """Create sample dataset for testing"""
    np.random.seed(42)

    data = {
        # Numerical features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),

        # Categorical features with some rare categories
        'country': np.random.choice(
            ['USA', 'UK', 'Canada', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Switzerland'],
            n_samples,
            p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.03, 0.02, 0.01, 0.005, 0.005]
        ),
        'product': np.random.choice(
            ['A', 'B', 'C', 'D', 'E', 'F'],
            n_samples,
            p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]
        ),
    }

    return pd.DataFrame(data)


def test_numerical_transformations():
    """Test numerical feature transformations"""
    print("=" * 60)
    print("Testing Numerical Transformations")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print("\nOriginal numerical features:")
    print(df[['age', 'income', 'credit_score']].describe())

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Transform numerical features
    numerical_cols = ['age', 'income', 'credit_score']
    df_transformed = fe.fit_transform_numerical(df, numerical_cols)

    print(f"\nTransformed data shape: {df_transformed.shape}")
    print(f"New features created: {df_transformed.shape[1] - df.shape[1]}")

    # Show new features
    print("\nNew feature columns:")
    new_cols = [col for col in df_transformed.columns if col not in df.columns]
    for col in new_cols:
        print(f"  - {col}")

    # Check transformations
    print("\n" + "-" * 60)
    print("Checking transformations:")
    print("-" * 60)

    for col in numerical_cols:
        print(f"\n{col.upper()}:")

        # Capped version
        capped_col = f"{col}_capped"
        print(f"  {capped_col}:")
        print(f"    Original range: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"    Capped range:   [{df_transformed[capped_col].min():.2f}, {df_transformed[capped_col].max():.2f}]")

        # Binned versions
        for n_bins in [10, 20]:
            binned_col = f"{col}_binned_{n_bins}"
            print(f"  {binned_col}:")
            print(f"    Unique bins: {df_transformed[binned_col].nunique()}")
            print(f"    Value counts: {dict(df_transformed[binned_col].value_counts().head(3))}")

    # Feature summary
    print("\n" + "=" * 60)
    print("Feature Summary:")
    print("=" * 60)
    print(fe.get_feature_summary().to_string(index=False))

    return fe, df_transformed


def test_categorical_transformations():
    """Test categorical feature grouping"""
    print("\n\n")
    print("=" * 60)
    print("Testing Categorical Transformations")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")

    # Initialize feature engineer
    config = FeatureTransformConfig(min_category_freq=0.05)
    fe = FeatureEngineer(config)

    # Transform categorical features
    categorical_cols = ['country', 'product']
    df_transformed = fe.fit_transform_categorical(df, categorical_cols)

    print(f"\nTransformed data shape: {df_transformed.shape}")

    # Show transformations
    print("\n" + "-" * 60)
    print("Checking categorical groupings:")
    print("-" * 60)

    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        print(f"  Original categories ({df[col].nunique()}):")
        original_counts = df[col].value_counts()
        for cat, count in original_counts.items():
            freq = count / len(df)
            marker = " [RARE]" if freq < config.min_category_freq else ""
            print(f"    {cat}: {count} ({freq:.1%}){marker}")

        grouped_col = f"{col}_grouped"
        if grouped_col in df_transformed.columns:
            print(f"\n  Grouped categories ({df_transformed[grouped_col].nunique()}):")
            grouped_counts = df_transformed[grouped_col].value_counts()
            for cat, count in grouped_counts.items():
                freq = count / len(df_transformed)
                print(f"    {cat}: {count} ({freq:.1%})")

    # Feature summary
    print("\n" + "=" * 60)
    print("Feature Summary:")
    print("=" * 60)
    print(fe.get_feature_summary().to_string(index=False))

    return fe, df_transformed


def test_full_pipeline():
    """Test complete feature engineering pipeline"""
    print("\n\n")
    print("=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)

    # Create train and test data
    df_train = create_sample_data(1000)
    df_test = create_sample_data(200)

    print(f"\nTrain data: {df_train.shape}")
    print(f"Test data: {df_test.shape}")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Fit and transform on training data
    numerical_cols = ['age', 'income', 'credit_score']
    categorical_cols = ['country', 'product']

    print("\nTransforming training data...")
    df_train_num = fe.fit_transform_numerical(df_train, numerical_cols)
    df_train_transformed = fe.fit_transform_categorical(df_train_num, categorical_cols)

    print(f"Training data transformed: {df_train_transformed.shape}")

    # Transform test data using fitted parameters
    print("\nTransforming test data...")
    df_test_num = fe.transform_numerical(df_test, numerical_cols)
    df_test_transformed = fe.transform_categorical(df_test_num, categorical_cols)

    print(f"Test data transformed: {df_test_transformed.shape}")

    # Get feature list
    print("\n" + "=" * 60)
    print("All Transformed Features:")
    print("=" * 60)
    all_features = fe.get_feature_list()
    for i, feat in enumerate(all_features, 1):
        print(f"{i:2d}. {feat}")

    print(f"\nTotal features: {len(all_features)}")
    print(f"Original features: {len(numerical_cols) + len(categorical_cols)}")
    print(f"New features created: {len(all_features) - len(numerical_cols) - len(categorical_cols)}")

    return fe, df_train_transformed, df_test_transformed


if __name__ == "__main__":
    # Run tests
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING TEST SUITE")
    print("=" * 60)

    # Test 1: Numerical transformations
    fe_num, df_num = test_numerical_transformations()

    # Test 2: Categorical transformations
    fe_cat, df_cat = test_categorical_transformations()

    # Test 3: Full pipeline
    fe_full, df_train, df_test = test_full_pipeline()

    print("\n\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
