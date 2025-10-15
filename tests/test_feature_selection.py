"""
Test Feature Selection Module
"""
import sys
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig


def create_classification_data(n_samples=500, n_features=20, n_informative=5):
    """Create synthetic classification dataset"""
    from sklearn.datasets import make_classification

    # Adjust redundant and repeated based on total features
    n_redundant = min(5, max(0, n_features - n_informative - 2))
    n_repeated = min(2, max(0, n_features - n_informative - n_redundant))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=2,
        random_state=42
    )

    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    return X_df, y_series, feature_names


def test_mutual_info_selection():
    """Test mutual information feature selection"""
    print("=" * 60)
    print("Test 1: Mutual Information Selection")
    print("=" * 60)

    X, y, feature_names = create_classification_data()

    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: {y.nunique()} classes")

    # Configure selector
    config = FeatureSelectionConfig(
        methods=['mutual_info'],
        top_k=10,
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    # Get selected features
    selected = fs.get_selected_features()
    print(f"\nSelected {len(selected)} features: {selected}")

    assert len(selected) == 10, "Should select top 10 features"
    print("\n✓ Mutual information selection passed!")

    return fs


def test_tree_importance_selection():
    """Test Random Forest importance selection"""
    print("\n\n" + "=" * 60)
    print("Test 2: Tree-based Importance Selection")
    print("=" * 60)

    X, y, feature_names = create_classification_data()

    config = FeatureSelectionConfig(
        methods=['tree_importance'],
        top_k=10,
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    selected = fs.get_selected_features()
    print(f"\nSelected {len(selected)} features: {selected}")

    assert len(selected) == 10
    print("\n✓ Tree importance selection passed!")

    return fs


def test_correlation_selection():
    """Test correlation-based selection"""
    print("\n\n" + "=" * 60)
    print("Test 3: Correlation-based Selection")
    print("=" * 60)

    X, y, feature_names = create_classification_data()

    config = FeatureSelectionConfig(
        methods=['correlation'],
        threshold=0.05,  # Select features with correlation > 0.05
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    selected = fs.get_selected_features()
    print(f"\nSelected {len(selected)} features with correlation > 0.05")
    print(f"Features: {selected[:5]}..." if len(selected) > 5 else f"Features: {selected}")

    print("\n✓ Correlation selection passed!")

    return fs


def test_multi_method_selection():
    """Test selection with multiple methods (ensemble)"""
    print("\n\n" + "=" * 60)
    print("Test 4: Multi-Method Selection (Ensemble)")
    print("=" * 60)

    X, y, feature_names = create_classification_data()

    config = FeatureSelectionConfig(
        methods=['mutual_info', 'tree_importance', 'correlation', 'statistical'],
        top_k=8,
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    # Print summary
    fs.print_summary(top_n=15)

    selected = fs.get_selected_features()
    print(f"\n{'='*60}")
    print(f"Final selected features: {len(selected)}")
    print('='*60)
    for i, feat in enumerate(selected, 1):
        print(f"  {i}. {feat}")

    assert len(selected) == 8
    print("\n✓ Multi-method selection passed!")

    return fs


def test_transform():
    """Test transformation (selecting columns)"""
    print("\n\n" + "=" * 60)
    print("Test 5: Transform (Select Columns)")
    print("=" * 60)

    X, y, feature_names = create_classification_data()

    print(f"Original data: {X.shape}")

    config = FeatureSelectionConfig(
        methods=['mutual_info'],
        top_k=5,
        task='classification'
    )

    fs = FeatureSelector(config)
    X_selected = fs.fit_transform(X, y)

    print(f"Transformed data: {X_selected.shape}")
    print(f"Selected columns: {list(X_selected.columns)}")

    assert X_selected.shape[1] == 5
    assert X_selected.shape[0] == X.shape[0]

    print("\n✓ Transform test passed!")

    return fs, X_selected


def test_feature_scores():
    """Test getting feature scores"""
    print("\n\n" + "=" * 60)
    print("Test 6: Feature Scores Retrieval")
    print("=" * 60)

    X, y, feature_names = create_classification_data(n_features=10)

    config = FeatureSelectionConfig(
        methods=['mutual_info', 'tree_importance'],
        top_k=5,
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    # Get aggregated scores
    agg_scores = fs.get_feature_scores()
    print(f"\nAggregated scores (top 5):")
    for feat, score in list(agg_scores.items())[:5]:
        print(f"  {feat}: {score:.4f}")

    # Get method-specific scores
    mi_scores = fs.get_feature_scores('mutual_info')
    print(f"\nMutual Information scores (top 5):")
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_mi[:5]:
        print(f"  {feat}: {score:.4f}")

    # Get top features
    top_5 = fs.get_top_features(k=5)
    print(f"\nTop 5 features: {top_5}")

    assert len(agg_scores) == 10
    assert len(top_5) == 5

    print("\n✓ Feature scores test passed!")

    return fs


def test_with_transformed_features():
    """Test feature selection on engineered features"""
    print("\n\n" + "=" * 60)
    print("Test 7: Feature Selection on Engineered Features")
    print("=" * 60)

    # Simulate engineered features (original + transformed)
    np.random.seed(42)
    n_samples = 300

    df = pd.DataFrame({
        # Original features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'score': np.random.normal(700, 100, n_samples),

        # Binary features (should be less important if truly random)
        'is_premium': np.random.choice([0, 1], n_samples),
        'has_loan': np.random.choice([0, 1], n_samples),

        # Engineered features (capped, binned)
        'age_capped': np.random.randint(18, 80, n_samples),
        'age_binned_10': np.random.randint(0, 10, n_samples),
        'income_capped': np.random.exponential(50000, n_samples),
        'income_binned_10': np.random.randint(0, 10, n_samples),

        # Target (correlated with age and income for realism)
    })

    # Create target with some correlation
    df['target'] = ((df['age'] > 40).astype(int) +
                    (df['income'] > 60000).astype(int)) % 2

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Features: {len(X.columns)}")
    print(f"Columns: {list(X.columns)}")

    config = FeatureSelectionConfig(
        methods=['mutual_info', 'tree_importance', 'correlation'],
        top_k=5,
        task='classification'
    )

    fs = FeatureSelector(config)
    fs.fit(X, y)

    fs.print_summary(top_n=10)

    selected = fs.get_selected_features()
    print(f"\n{'='*60}")
    print(f"Top 5 selected features for modeling:")
    print('='*60)
    for i, feat in enumerate(selected, 1):
        print(f"  {i}. {feat}")

    assert len(selected) == 5
    print("\n✓ Engineered features test passed!")

    return fs


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FEATURE SELECTION TEST SUITE")
    print("=" * 60)

    # Run tests
    fs1 = test_mutual_info_selection()
    fs2 = test_tree_importance_selection()
    fs3 = test_correlation_selection()
    fs4 = test_multi_method_selection()
    fs5, X_sel = test_transform()
    fs6 = test_feature_scores()
    fs7 = test_with_transformed_features()

    print("\n\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nFeature Selection Methods Available:")
    print("  ✓ Mutual Information")
    print("  ✓ Random Forest Importance")
    print("  ✓ Correlation-based")
    print("  ✓ Statistical tests (ANOVA F-test)")
    print("  ✓ Multi-method ensemble")
