"""
Feature Selection Module
Ranks and selects the best features for modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import spearmanr, pearsonr


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    # Selection method
    methods: List[str] = None  # ['mutual_info', 'tree_importance', 'correlation', 'statistical']

    # Selection criteria
    top_k: Optional[int] = None  # Select top K features
    threshold: Optional[float] = None  # Select features above threshold

    # Redundancy removal
    max_correlation: float = 0.95  # Remove features correlated > this threshold

    # Task type
    task: str = 'classification'  # 'classification' or 'regression'

    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_random_state: int = 42

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['mutual_info', 'tree_importance', 'correlation']


class FeatureSelector:
    """
    Selects best features for modeling using multiple methods

    Methods:
    - Mutual Information: Measures dependency between features and target
    - Tree-based Importance: Random Forest feature importance
    - Correlation: Correlation coefficient with target
    - Statistical: ANOVA F-test (classification) or F-regression (regression)
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.feature_scores = {}  # Method -> {feature: score}
        self.feature_rankings = {}  # Method -> [features sorted by importance]
        self.selected_features = []
        self.feature_summary = None
        self.removed_redundant = []  # Features removed due to correlation
        self.X_numeric = None  # Store for correlation calculation

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureSelector':
        """
        Compute feature importance scores using multiple methods

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names (uses X.columns if None)

        Returns:
            self
        """
        if feature_names is None:
            feature_names = X.columns.tolist()

        # Ensure X is numeric
        X_numeric = self._prepare_features(X)

        # Store for correlation calculation
        self.X_numeric = X_numeric
        self.feature_names = feature_names

        print(f"\n{'='*60}")
        print("FEATURE SELECTION")
        print('='*60)
        print(f"Task: {self.config.task}")
        print(f"Features: {len(feature_names)}")
        print(f"Samples: {len(X)}")
        print(f"Methods: {', '.join(self.config.methods)}")
        print(f"Redundancy removal: max_correlation={self.config.max_correlation}")

        # Compute scores with each method
        for method in self.config.methods:
            print(f"\nComputing {method} scores...")
            try:
                if method == 'mutual_info':
                    scores = self._mutual_info_scores(X_numeric, y)
                elif method == 'tree_importance':
                    scores = self._tree_importance_scores(X_numeric, y)
                elif method == 'correlation':
                    scores = self._correlation_scores(X_numeric, y)
                elif method == 'statistical':
                    scores = self._statistical_scores(X_numeric, y)
                else:
                    print(f"  Warning: Unknown method '{method}', skipping")
                    continue

                self.feature_scores[method] = dict(zip(feature_names, scores))

                # Rank features
                ranked_features = sorted(
                    zip(feature_names, scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.feature_rankings[method] = [f[0] for f in ranked_features]

                print(f"  ✓ Computed {method} scores")

            except Exception as e:
                print(f"  ✗ Error computing {method}: {e}")

        # Select features
        self._select_features(feature_names)

        return self

    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Convert features to numeric array, handling categoricals"""
        X_numeric = X.copy()

        # Convert categorical to numeric codes
        for col in X_numeric.select_dtypes(include=['object', 'category']).columns:
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes

        # Fill NaN with median
        X_numeric = X_numeric.fillna(X_numeric.median())

        return X_numeric.values

    def _mutual_info_scores(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """Compute mutual information scores"""
        if self.config.task == 'classification':
            scores = mutual_info_classif(X, y, random_state=self.config.rf_random_state)
        else:
            scores = mutual_info_regression(X, y, random_state=self.config.rf_random_state)
        return scores

    def _tree_importance_scores(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """Compute Random Forest feature importance"""
        if self.config.task == 'classification':
            model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                random_state=self.config.rf_random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                random_state=self.config.rf_random_state,
                n_jobs=-1
            )

        model.fit(X, y)
        return model.feature_importances_

    def _correlation_scores(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """Compute absolute correlation with target"""
        scores = []
        for i in range(X.shape[1]):
            # Use Spearman for robustness
            corr, _ = spearmanr(X[:, i], y)
            scores.append(abs(corr))
        return np.array(scores)

    def _statistical_scores(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """Compute statistical test scores (ANOVA F-test)"""
        if self.config.task == 'classification':
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        # Replace NaN with 0
        scores = np.nan_to_num(scores, nan=0.0)
        return scores

    def _select_features(self, feature_names: List[str]):
        """Select features based on criteria with redundancy removal"""
        print(f"\n{'='*60}")
        print("FEATURE SELECTION CRITERIA")
        print('='*60)

        # Aggregate scores (average across methods)
        aggregated_scores = {}
        for feature in feature_names:
            scores = []
            for method in self.feature_scores:
                if feature in self.feature_scores[method]:
                    scores.append(self.feature_scores[method][feature])

            if scores:
                # Normalize scores per method, then average
                aggregated_scores[feature] = np.mean(scores)

        # Sort by aggregated score
        sorted_features = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Determine target count based on criteria
        if self.config.top_k is not None:
            target_k = self.config.top_k
            print(f"Criterion: Top {self.config.top_k} features (with redundancy removal)")
        elif self.config.threshold is not None:
            target_k = len([f for f in sorted_features if f[1] >= self.config.threshold])
            print(f"Criterion: Score >= {self.config.threshold} (with redundancy removal)")
        else:
            target_k = len(sorted_features)
            print(f"Criterion: All features (ranked, with redundancy removal)")

        # Select features while removing redundancy
        self.selected_features = []
        self.removed_redundant = []
        redundancy_details = {}  # feature -> (corr, correlated_with)

        for feature, score in sorted_features:
            # Stop if we have enough features
            if self.config.top_k is not None and len(self.selected_features) >= self.config.top_k:
                break

            # Check threshold
            if self.config.threshold is not None and score < self.config.threshold:
                break

            # Check correlation with already selected features
            if self.selected_features and self.config.max_correlation < 1.0:
                feature_idx = self.feature_names.index(feature)
                is_redundant = False

                for selected_feature in self.selected_features:
                    selected_idx = self.feature_names.index(selected_feature)

                    # Calculate correlation
                    corr = np.corrcoef(
                        self.X_numeric[:, feature_idx],
                        self.X_numeric[:, selected_idx]
                    )[0, 1]

                    if abs(corr) > self.config.max_correlation:
                        is_redundant = True
                        self.removed_redundant.append(feature)
                        redundancy_details[feature] = (abs(corr), selected_feature)
                        break

                if is_redundant:
                    continue

            # Add feature to selected
            self.selected_features.append(feature)

        print(f"Selected: {len(self.selected_features)} / {len(feature_names)} features")
        if self.removed_redundant:
            print(f"Removed as redundant: {len(self.removed_redundant)} features")

        # Create summary
        self.feature_summary = pd.DataFrame([
            {
                'feature': feature,
                'aggregated_score': aggregated_scores.get(feature, 0),
                'selected': feature in self.selected_features,
                'redundant': feature in self.removed_redundant,
                'correlated_with': redundancy_details.get(feature, (None, None))[1] if feature in redundancy_details else None,
                'correlation': redundancy_details.get(feature, (None, None))[0] if feature in redundancy_details else None,
                **{f'{method}_score': self.feature_scores[method].get(feature, 0)
                   for method in self.feature_scores}
            }
            for feature in feature_names
        ]).sort_values('aggregated_score', ascending=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the chosen features"""
        if not self.selected_features:
            raise ValueError("No features selected. Call fit() first.")

        # Select features that exist in X
        available_features = [f for f in self.selected_features if f in X.columns]

        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            print(f"Warning: {len(missing)} selected features not found in data: {missing}")

        return X[available_features]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_feature_scores(self, method: Optional[str] = None) -> Dict[str, float]:
        """Get feature scores for a specific method or aggregated"""
        if method is None:
            # Return aggregated scores
            return dict(zip(
                self.feature_summary['feature'],
                self.feature_summary['aggregated_score']
            ))
        elif method in self.feature_scores:
            return self.feature_scores[method]
        else:
            raise ValueError(f"Method '{method}' not found. Available: {list(self.feature_scores.keys())}")

    def get_top_features(self, k: int = 10, method: Optional[str] = None) -> List[str]:
        """Get top K features by importance"""
        if method is None:
            # Use aggregated ranking
            return self.feature_summary['feature'].head(k).tolist()
        elif method in self.feature_rankings:
            return self.feature_rankings[method][:k]
        else:
            raise ValueError(f"Method '{method}' not found")

    def get_selected_features(self) -> List[str]:
        """Get list of selected features"""
        return self.selected_features.copy()

    def print_summary(self, top_n: int = 20):
        """Print feature selection summary"""
        print(f"\n{'='*60}")
        print("FEATURE SELECTION SUMMARY")
        print('='*60)

        if self.feature_summary is None:
            print("No features selected yet. Call fit() first.")
            return

        print(f"\nTotal features evaluated: {len(self.feature_summary)}")
        print(f"Features selected: {len(self.selected_features)}")
        print(f"Features removed as redundant: {len(self.removed_redundant)}")
        print(f"Selection rate: {len(self.selected_features) / len(self.feature_summary):.1%}")

        print(f"\n{'='*60}")
        print(f"Top {top_n} Features (by aggregated score):")
        print('='*60)

        display_cols = ['feature', 'aggregated_score', 'selected', 'redundant'] + \
                      [f'{m}_score' for m in self.feature_scores]

        summary_display = self.feature_summary[display_cols].head(top_n)
        print(summary_display.to_string(index=False, float_format='%.4f'))

        # Show redundant features details
        if self.removed_redundant:
            print(f"\n{'='*60}")
            print("Redundant Features (removed due to high correlation):")
            print('='*60)
            redundant_df = self.feature_summary[
                self.feature_summary['redundant'] == True
            ][['feature', 'aggregated_score', 'correlated_with', 'correlation']].head(20)
            print(redundant_df.to_string(index=False, float_format='%.4f'))

        if len(self.selected_features) < len(self.feature_summary):
            print(f"\n{'='*60}")
            print("Bottom 5 Features:")
            print('='*60)
            bottom_5 = self.feature_summary[display_cols].tail(5)
            print(bottom_5.to_string(index=False, float_format='%.4f'))
