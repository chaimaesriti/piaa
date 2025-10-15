"""
Test redundancy removal in feature selection
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.features.feature_selection import FeatureSelector, FeatureSelectionConfig


def test_redundancy_removal():
    """Test that highly correlated features are removed"""
    print("="*60)
    print("Testing Redundancy Removal")
    print("="*60)

    # Create sample data with correlated features
    np.random.seed(42)
    n_samples = 1000

    # Base feature
    age = np.random.randint(18, 80, n_samples)

    # Create correlated features (like age, age_capped, age_binned_10, age_binned_20)
    age_capped = np.clip(age, 20, 70)  # Very correlated with age
    age_plus_noise = age + np.random.normal(0, 0.5, n_samples)  # Almost identical

    # Independent features
    income = np.random.randint(20000, 150000, n_samples)
    score = np.random.randint(300, 850, n_samples)

    # Target (correlated with age and income)
    target = (age * 0.5 + income * 0.00001 + np.random.normal(0, 5, n_samples)) > 40
    target = target.astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'age_capped': age_capped,
        'age_plus_noise': age_plus_noise,
        'income': income,
        'score': score,
        'target': target
    })

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check correlations
    print("\nCorrelations between age features:")
    print(f"age vs age_capped: {df['age'].corr(df['age_capped']):.4f}")
    print(f"age vs age_plus_noise: {df['age'].corr(df['age_plus_noise']):.4f}")
    print(f"age_capped vs age_plus_noise: {df['age_capped'].corr(df['age_plus_noise']):.4f}")

    # Test 1: Without redundancy removal (max_correlation=1.0)
    print("\n" + "="*60)
    print("Test 1: Without redundancy removal (max_correlation=1.0)")
    print("="*60)

    X = df.drop(columns=['target'])
    y = df['target']

    config_no_removal = FeatureSelectionConfig(
        methods=['mutual_info', 'correlation'],
        top_k=3,
        max_correlation=1.0,  # Disable redundancy removal
        task='classification'
    )

    fs_no_removal = FeatureSelector(config_no_removal)
    fs_no_removal.fit(X, y)

    selected_no_removal = fs_no_removal.get_selected_features()
    print(f"\nSelected features: {selected_no_removal}")
    print(f"Expected: Should include correlated age features")

    # Test 2: With redundancy removal (max_correlation=0.95)
    print("\n" + "="*60)
    print("Test 2: With redundancy removal (max_correlation=0.95)")
    print("="*60)

    config_with_removal = FeatureSelectionConfig(
        methods=['mutual_info', 'correlation'],
        top_k=3,
        max_correlation=0.95,  # Enable redundancy removal
        task='classification'
    )

    fs_with_removal = FeatureSelector(config_with_removal)
    fs_with_removal.fit(X, y)
    fs_with_removal.print_summary(top_n=10)

    selected_with_removal = fs_with_removal.get_selected_features()
    print(f"\nSelected features: {selected_with_removal}")
    print(f"Removed as redundant: {fs_with_removal.removed_redundant}")

    # Verify
    print("\n" + "="*60)
    print("Verification")
    print("="*60)

    # Should have exactly 3 features
    assert len(selected_with_removal) == 3, f"Expected 3 features, got {len(selected_with_removal)}"
    print("✓ Selected exactly 3 features")

    # Should have removed some redundant features
    assert len(fs_with_removal.removed_redundant) > 0, "Expected some redundant features to be removed"
    print(f"✓ Removed {len(fs_with_removal.removed_redundant)} redundant features")

    # Should not have multiple age variants
    age_variants = [f for f in selected_with_removal if 'age' in f.lower()]
    assert len(age_variants) <= 1, f"Expected at most 1 age variant, got {age_variants}"
    print(f"✓ Only {len(age_variants)} age variant(s) selected (avoiding redundancy)")

    # Should include diverse features
    print(f"✓ Selected diverse features: {selected_with_removal}")

    print("\n" + "="*60)
    print("✅ REDUNDANCY REMOVAL TEST PASSED")
    print("="*60)


if __name__ == '__main__':
    test_redundancy_removal()
