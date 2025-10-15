"""
Feature Quality Filter
Filters out features that are unsuitable for modeling
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FeatureFilterConfig:
    """Configuration for feature filtering"""
    # Missingness threshold
    max_missing_rate: float = 0.90  # Remove features with >90% missing

    # Cardinality thresholds
    max_cardinality_numeric: int = 1000  # Max unique values for numeric
    max_cardinality_categorical: int = 100  # Max unique values for categorical

    # Additional filters
    min_variance: float = 0.0  # Remove zero-variance features
    max_correlation: float = 0.99  # Remove highly correlated features


class FeatureFilter:
    """
    Filters features based on quality criteria

    Removes:
    - Features with high missingness (>90% by default)
    - Numerical features with high cardinality (>1000 unique values)
    - Categorical features with high cardinality (>100 unique values)
    - Zero or near-zero variance features
    """

    def __init__(self, config: Optional[FeatureFilterConfig] = None):
        self.config = config or FeatureFilterConfig()
        self.filter_report = {}
        self.removed_features = {}
        self.kept_features = []

    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> 'FeatureFilter':
        """
        Analyze features and determine which to keep

        Args:
            df: Input dataframe
            numerical_cols: List of numerical columns (auto-detected if None)
            categorical_cols: List of categorical columns (auto-detected if None)
            target_col: Target column to exclude from filtering

        Returns:
            self
        """
        # Auto-detect column types if not provided
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target from filtering
        if target_col:
            numerical_cols = [c for c in numerical_cols if c != target_col]
            categorical_cols = [c for c in categorical_cols if c != target_col]

        all_features = numerical_cols + categorical_cols

        # Initialize tracking
        self.removed_features = {
            'high_missingness': [],
            'high_cardinality': [],
            'zero_variance': [],
            'all': []
        }

        self.filter_report = {}

        # Check each feature
        for col in all_features:
            col_report = {
                'type': 'numerical' if col in numerical_cols else 'categorical',
                'missing_rate': df[col].isna().sum() / len(df),
                'cardinality': df[col].nunique(),
                'variance': df[col].var() if col in numerical_cols and pd.api.types.is_numeric_dtype(df[col]) else None,
                'kept': True,
                'removal_reason': None
            }

            # Check 1: High missingness
            if col_report['missing_rate'] > self.config.max_missing_rate:
                col_report['kept'] = False
                col_report['removal_reason'] = 'high_missingness'
                self.removed_features['high_missingness'].append(col)
                self.removed_features['all'].append(col)

            # Check 2: High cardinality
            elif col in numerical_cols and col_report['cardinality'] >= self.config.max_cardinality_numeric:
                col_report['kept'] = False
                col_report['removal_reason'] = 'high_cardinality'
                self.removed_features['high_cardinality'].append(col)
                self.removed_features['all'].append(col)

            elif col in categorical_cols and col_report['cardinality'] >= self.config.max_cardinality_categorical:
                col_report['kept'] = False
                col_report['removal_reason'] = 'high_cardinality'
                self.removed_features['high_cardinality'].append(col)
                self.removed_features['all'].append(col)

            # Check 3: Zero variance (numerical only)
            elif col in numerical_cols and col_report['variance'] is not None:
                if col_report['variance'] <= self.config.min_variance:
                    col_report['kept'] = False
                    col_report['removal_reason'] = 'zero_variance'
                    self.removed_features['zero_variance'].append(col)
                    self.removed_features['all'].append(col)

            self.filter_report[col] = col_report

        # Store kept features
        self.kept_features = [col for col in all_features if self.filter_report[col]['kept']]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove filtered features from dataframe

        Args:
            df: Input dataframe

        Returns:
            DataFrame with only kept features
        """
        if not self.kept_features:
            raise ValueError("Filter not fitted. Call fit() first.")

        # Keep only the features that passed filtering
        cols_to_keep = [col for col in self.kept_features if col in df.columns]

        # Also keep any columns not seen during fit (e.g., target)
        other_cols = [col for col in df.columns if col not in self.filter_report]

        return df[cols_to_keep + other_cols]

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, numerical_cols, categorical_cols, target_col)
        return self.transform(df)

    def get_filter_summary(self) -> pd.DataFrame:
        """Get summary of filtering results"""
        summary = []
        for col, report in self.filter_report.items():
            summary.append({
                'feature': col,
                'type': report['type'],
                'missing_rate': report['missing_rate'],
                'cardinality': report['cardinality'],
                'variance': report['variance'],
                'kept': report['kept'],
                'removal_reason': report['removal_reason'] or 'N/A'
            })
        return pd.DataFrame(summary)

    def get_removal_stats(self) -> Dict:
        """Get statistics on removed features"""
        total_features = len(self.filter_report)
        return {
            'total_features': total_features,
            'kept_features': len(self.kept_features),
            'removed_features': len(self.removed_features['all']),
            'removed_high_missingness': len(self.removed_features['high_missingness']),
            'removed_high_cardinality': len(self.removed_features['high_cardinality']),
            'removed_zero_variance': len(self.removed_features['zero_variance']),
            'kept_rate': len(self.kept_features) / total_features if total_features > 0 else 0
        }

    def print_summary(self):
        """Print human-readable summary"""
        stats = self.get_removal_stats()

        print("=" * 60)
        print("FEATURE FILTER SUMMARY")
        print("=" * 60)
        print(f"Total features:            {stats['total_features']}")
        print(f"Kept features:             {stats['kept_features']} ({stats['kept_rate']:.1%})")
        print(f"Removed features:          {stats['removed_features']}")
        print()
        print("Removal reasons:")
        print(f"  - High missingness (>{self.config.max_missing_rate:.0%}): {stats['removed_high_missingness']}")
        print(f"  - High cardinality:                {stats['removed_high_cardinality']}")
        print(f"  - Zero variance:                   {stats['removed_zero_variance']}")
        print()

        if self.removed_features['all']:
            print("Removed features:")
            for reason, features in self.removed_features.items():
                if reason != 'all' and features:
                    print(f"\n  {reason}:")
                    for feat in features:
                        report = self.filter_report[feat]
                        if reason == 'high_missingness':
                            print(f"    - {feat} (missing: {report['missing_rate']:.1%})")
                        elif reason == 'high_cardinality':
                            print(f"    - {feat} (cardinality: {report['cardinality']})")
                        elif reason == 'zero_variance':
                            print(f"    - {feat} (variance: {report['variance']})")

        print("\n" + "=" * 60)

    def get_kept_features(self) -> List[str]:
        """Get list of features that passed filtering"""
        return self.kept_features.copy()

    def get_removed_features(self) -> List[str]:
        """Get list of features that were removed"""
        return self.removed_features['all'].copy()
