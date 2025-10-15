"""
Test complete pipeline: Feature Engineering + Filtering
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, FeatureTransformConfig
from src.features.feature_filter import FeatureFilter, FeatureFilterConfig


def create_sample_data():
    """Create realistic sample data"""
    np.random.seed(42)
    n = 500

    # Good features
    age = np.random.randint(18, 80, n)
    income = np.random.exponential(50000, n)
    credit_score = np.random.normal(700, 100, n)
    country = np.random.choice(['USA', 'UK', 'Canada', 'France'], n)
    product = np.random.choice(['A', 'B', 'C'], n)

    # Bad features
    # High missingness
    high_missing = np.random.randn(n)
    high_missing[np.random.random(n) > 0.08] = np.nan  # 92% missing

    # High cardinality ID
    user_id = np.arange(n)

    # Target
    target = np.random.randint(0, 2, n)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'country': country,
        'product': product,
        'high_missing_col': high_missing,
        'user_id': user_id,
        'target': target
    })

    return df


def test_full_pipeline():
    """Test: Transform -> Filter -> Model-ready"""
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    # 1. Load data
    df = create_sample_data()
    print(f"\nStep 1: Loaded data")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # 2. Feature Engineering
    print(f"\n{'='*60}")
    print("Step 2: Feature Engineering")
    print('='*60)

    numerical_cols = ['age', 'income', 'credit_score', 'high_missing_col', 'user_id']
    categorical_cols = ['country', 'product']

    fe_config = FeatureTransformConfig(
        cap_percentiles=(1, 99),
        n_bins_options=[10, 20],
        min_category_freq=0.01
    )

    fe = FeatureEngineer(fe_config)
    df_transformed = fe.fit_transform_numerical(df, numerical_cols)
    df_transformed = fe.fit_transform_categorical(df_transformed, categorical_cols)

    print(f"  Transformed shape: {df_transformed.shape}")
    print(f"  New features: {df_transformed.shape[1] - df.shape[1]}")

    # 3. Feature Filtering
    print(f"\n{'='*60}")
    print("Step 3: Feature Quality Filtering")
    print('='*60)

    filter_config = FeatureFilterConfig(
        max_missing_rate=0.90,
        max_cardinality_numeric=1000,
        max_cardinality_categorical=100
    )

    ff = FeatureFilter(filter_config)
    df_filtered = ff.fit_transform(df_transformed, target_col='target')

    print(f"  Filtered shape: {df_filtered.shape}")
    print(f"  Features removed: {len(ff.removed_features['all'])}")

    ff.print_summary()

    # 4. Model-ready features
    print(f"\n{'='*60}")
    print("Step 4: Model-Ready Features")
    print('='*60)

    X_cols = [col for col in df_filtered.columns if col != 'target']
    y_col = 'target'

    print(f"  Features for modeling: {len(X_cols)}")
    print(f"  Target: {y_col}")
    print(f"\n  Feature list:")
    for i, col in enumerate(X_cols, 1):
        print(f"    {i:2d}. {col}")

    # Check data quality
    print(f"\n{'='*60}")
    print("Step 5: Data Quality Check")
    print('='*60)

    X = df_filtered[X_cols]
    y = df_filtered[y_col]

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Missing values in X: {X.isna().sum().sum()}")
    print(f"  Missing values in y: {y.isna().sum()}")

    print(f"\n  Max cardinality:")
    for col in X_cols[:5]:  # Show first 5
        cardinality = X[col].nunique()
        print(f"    {col}: {cardinality}")

    print("\n✓ Pipeline complete! Data is ready for modeling.")

    return df_filtered, X_cols, y_col


if __name__ == "__main__":
    df_filtered, features, target = test_full_pipeline()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final dataset: {df_filtered.shape}")
    print(f"Features: {len(features)}")
    print(f"Target: {target}")
    print(f"\nReady for model training!")
