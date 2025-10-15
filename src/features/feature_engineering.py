"""
Feature Engineering Module
Transforms numerical and categorical features with tracking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FeatureTransformConfig:
    """Configuration for feature transformations"""
    # Numerical transformations
    cap_percentiles: Tuple[float, float] = (1, 99)  # Percentiles for capping
    n_bins_options: List[int] = field(default_factory=lambda: [10, 20])

    # Categorical transformations
    min_category_freq: float = 0.01  # Minimum frequency to keep category separate


class FeatureEngineer:
    """
    Creates transformed features for numerical and categorical data

    For numerical features:
    - feature_capped: capped at percentiles
    - feature_binned_10: discretized into 10 bins
    - feature_binned_20: discretized into 20 bins

    For categorical features:
    - Groups rare categories into cat1_cat2_cat3_other

    Binary features (2 unique values) are kept as-is without transformation.
    """

    def __init__(self, config: Optional[FeatureTransformConfig] = None):
        self.config = config or FeatureTransformConfig()
        self.transform_stats = {}  # Store transformation parameters
        self.feature_mapping = {}  # Original -> transformed features
        self.binary_features = []  # Track binary features
        self.target_col = None  # Target column to exclude

    def detect_binary_features(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """
        Detect binary features (features with exactly 2 unique values)

        Args:
            df: Input dataframe
            columns: List of columns to check

        Returns:
            List of binary feature names
        """
        binary_features = []
        for col in columns:
            n_unique = df[col].nunique()
            if n_unique == 2:
                binary_features.append(col)
                unique_vals = df[col].dropna().unique()
                self.transform_stats[col] = {
                    'type': 'binary',
                    'unique_values': unique_vals.tolist()
                }
        return binary_features

    def set_target(self, target_col: str):
        """Set target column to exclude from transformations"""
        self.target_col = target_col

    def fit_transform_numerical(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create transformed versions of numerical features

        Args:
            df: Input dataframe
            numerical_cols: List of numerical column names
            target_col: Target column to exclude from transformations

        Returns:
            DataFrame with original + transformed features
        """
        # Update target if provided
        if target_col:
            self.target_col = target_col

        df_transformed = df.copy()

        # Detect binary features
        binary_cols = self.detect_binary_features(df, numerical_cols)
        self.binary_features.extend(binary_cols)

        # Filter out target and binary features
        cols_to_transform = [
            col for col in numerical_cols
            if col != self.target_col and col not in binary_cols
        ]

        # Log skipped features
        if binary_cols:
            print(f"\nℹ  Binary features detected (skipping transformation): {binary_cols}")
        if self.target_col and self.target_col in numerical_cols:
            print(f"ℹ  Target column skipped: {self.target_col}")

        # Mark binary features as-is (no transformation)
        for col in binary_cols:
            self.feature_mapping[col] = [col]  # Keep as-is

        for col in cols_to_transform:
            # Store original feature mapping
            self.feature_mapping[col] = []

            # 1. Capped version
            lower, upper = np.percentile(
                df[col].dropna(),
                self.config.cap_percentiles
            )
            capped_col = f"{col}_capped"
            df_transformed[capped_col] = df[col].clip(lower, upper)
            self.feature_mapping[col].append(capped_col)

            # Store stats for reproduction
            self.transform_stats[capped_col] = {
                'type': 'capped',
                'lower': lower,
                'upper': upper
            }

            # 2. Binned versions
            for n_bins in self.config.n_bins_options:
                binned_col = f"{col}_binned_{n_bins}"

                # Check if we have enough valid values for binning
                valid_values = df[col].dropna()
                if len(valid_values) >= n_bins:
                    try:
                        df_transformed[binned_col], bins = pd.cut(
                            df[col],
                            bins=n_bins,
                            retbins=True,
                            labels=False,
                            duplicates='drop'
                        )
                        self.feature_mapping[col].append(binned_col)

                        # Store stats
                        self.transform_stats[binned_col] = {
                            'type': 'binned',
                            'n_bins': n_bins,
                            'bin_edges': bins.tolist()
                        }
                    except Exception as e:
                        # Skip binning if it fails (e.g., too many missing values)
                        print(f"Warning: Could not bin {col} with {n_bins} bins: {e}")
                else:
                    print(f"Warning: Skipping binning for {col} (insufficient valid values: {len(valid_values)} < {n_bins})")

        return df_transformed

    def transform_numerical(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply fitted transformations to new data"""
        df_transformed = df.copy()

        for col in numerical_cols:
            if col not in self.feature_mapping:
                raise ValueError(f"Feature {col} not fitted. Call fit_transform first.")

            for transformed_col in self.feature_mapping[col]:
                stats = self.transform_stats[transformed_col]

                if stats['type'] == 'capped':
                    df_transformed[transformed_col] = df[col].clip(
                        stats['lower'],
                        stats['upper']
                    )
                elif stats['type'] == 'binned':
                    df_transformed[transformed_col] = pd.cut(
                        df[col],
                        bins=stats['bin_edges'],
                        labels=False,
                        include_lowest=True,
                        duplicates='drop'
                    )

        return df_transformed

    def fit_transform_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Group rare categories together

        Categories with frequency < min_category_freq are grouped as:
        cat1_cat2_cat3_other

        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names
            target_col: Target column to exclude from transformations

        Returns:
            DataFrame with original + transformed features
        """
        # Update target if provided
        if target_col:
            self.target_col = target_col

        df_transformed = df.copy()

        # Detect binary features
        binary_cols = self.detect_binary_features(df, categorical_cols)
        self.binary_features.extend(binary_cols)

        # Filter out target and binary features
        cols_to_transform = [
            col for col in categorical_cols
            if col != self.target_col and col not in binary_cols
        ]

        # Log skipped features
        if binary_cols:
            print(f"\nℹ  Binary features detected (skipping transformation): {binary_cols}")
        if self.target_col and self.target_col in categorical_cols:
            print(f"ℹ  Target column skipped: {self.target_col}")

        # Mark binary features as-is (no transformation)
        for col in binary_cols:
            if col not in self.feature_mapping:
                self.feature_mapping[col] = [col]  # Keep as-is

        for col in cols_to_transform:
            # Calculate frequency of each category
            freq = df[col].value_counts(normalize=True)

            # Find rare categories
            rare_categories = freq[freq < self.config.min_category_freq].index.tolist()

            if rare_categories:
                # Create grouped column name
                grouped_col = f"{col}_grouped"

                # Create mapping
                category_mapping = {cat: cat for cat in freq.index}

                # Group rare categories
                other_label = "_".join(rare_categories[:3]) + "_other" if len(rare_categories) <= 3 else \
                              "_".join(rare_categories[:2]) + "_other"

                for rare_cat in rare_categories:
                    category_mapping[rare_cat] = other_label

                # Apply mapping
                df_transformed[grouped_col] = df[col].map(category_mapping)

                # Store mapping
                self.feature_mapping[col] = [grouped_col]
                self.transform_stats[grouped_col] = {
                    'type': 'categorical_grouped',
                    'mapping': category_mapping,
                    'rare_categories': rare_categories,
                    'min_freq': self.config.min_category_freq
                }
            else:
                # No grouping needed
                self.feature_mapping[col] = [col]

        return df_transformed

    def transform_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Apply fitted categorical transformations to new data"""
        df_transformed = df.copy()

        for col in categorical_cols:
            if col not in self.feature_mapping:
                raise ValueError(f"Feature {col} not fitted. Call fit_transform first.")

            for transformed_col in self.feature_mapping[col]:
                if transformed_col != col:  # If grouping was applied
                    stats = self.transform_stats[transformed_col]
                    mapping = stats['mapping']

                    # Apply mapping with default for unseen categories
                    default_label = list(stats['rare_categories'])[0] + "_other" \
                                   if stats['rare_categories'] else 'unknown'
                    df_transformed[transformed_col] = df[col].map(
                        lambda x: mapping.get(x, default_label)
                    )

        return df_transformed

    def get_feature_list(self) -> List[str]:
        """Get list of all transformed features"""
        all_features = []
        for original, transformed in self.feature_mapping.items():
            all_features.extend(transformed)
        return all_features

    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all transformations"""
        summary = []
        for original, transformed in self.feature_mapping.items():
            for trans_col in transformed:
                stats = self.transform_stats.get(trans_col, {})
                summary.append({
                    'original_feature': original,
                    'transformed_feature': trans_col,
                    'transformation_type': stats.get('type', 'none')
                })
        return pd.DataFrame(summary)
