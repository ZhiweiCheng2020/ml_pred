"""
Feature selection utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Handle feature selection with multiple methods"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureSelector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.selection_config = config['feature_selection']

        self.selector = None
        self.selected_features = None
        self.feature_importance = None
        self.is_fitted = False

    def select_features(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        feature_names: list,
                        model_name: str,
                        apply_selection: bool = True) -> Tuple[np.ndarray, list]:
        """
        Select features based on configuration and model type

        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names
            model_name: Name of the model
            apply_selection: Whether to apply feature selection

        Returns:
            Tuple of (selected features array, selected feature names)
        """
        # Models that don't need feature selection
        no_selection_models = ['xgboost', 'lightgbm', 'random_forest', 'catboost', 'tabnet']

        if not apply_selection or model_name in no_selection_models:
            logger.info(f"Skipping feature selection for {model_name}")
            return X, feature_names

        method = self.selection_config['method']

        if method == 'none':
            return X, feature_names

        if method == 'auto':
            # Automatically decide based on model type
            if model_name in ['ridge', 'elasticnet', 'sparse_nn']:
                method = 'smart_pca' if X.shape[1] > 100 else 'none'
            else:
                method = 'none'

        logger.info(f"Applying {method} feature selection for {model_name}")

        if method == 'smart_pca':
            X_selected, selected_names = self._apply_pca(X, y, feature_names)
        elif method == 'tree_based':
            X_selected, selected_names = self._apply_tree_based(X, y, feature_names)
        else:
            X_selected, selected_names = X, feature_names

        logger.info(f"Selected {len(selected_names)} features from {len(feature_names)}")

        return X_selected, selected_names

    def _apply_pca(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Apply PCA for dimensionality reduction

        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names

        Returns:
            Tuple of (transformed features, component names)
        """
        variance_threshold = self.selection_config.get('pca_variance_threshold', 0.95)

        # Determine number of components
        n_components = min(X.shape[0], X.shape[1])

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        # Find number of components for desired variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_selected = np.argmax(cumsum_variance >= variance_threshold) + 1

        # Apply PCA with selected components
        self.selector = PCA(n_components=n_components_selected, random_state=42)
        X_transformed = self.selector.fit_transform(X)

        # Create component names
        component_names = [f"PC{i+1}" for i in range(n_components_selected)]

        # Store feature importance (loadings)
        self.feature_importance = pd.DataFrame(
            self.selector.components_[:n_components_selected].T,
            index=feature_names,
            columns=component_names
        )

        logger.info(f"PCA: {n_components_selected} components explain "
                   f"{cumsum_variance[n_components_selected-1]:.2%} variance")

        return X_transformed, component_names

    def _apply_tree_based(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Apply tree-based feature selection

        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names

        Returns:
            Tuple of (selected features, selected feature names)
        """
        top_k = self.selection_config.get('tree_based_top_k', 100)
        estimator_type = self.selection_config.get('tree_based_estimator', 'random_forest')

        # Choose estimator
        if estimator_type == 'xgboost':
            estimator = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        # Fit estimator
        estimator.fit(X, y)

        # Get feature importance
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        else:
            importances = np.abs(estimator.coef_)

        # Select top k features
        top_k = min(top_k, X.shape[1])
        indices = np.argsort(importances)[-top_k:][::-1]

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Select features
        X_selected = X[:, indices]
        selected_names = [feature_names[i] for i in indices]

        logger.info(f"Tree-based selection: {top_k} features selected")

        return X_selected, selected_names

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> 'FeatureSelector':
        """
        Fit feature selector

        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names

        Returns:
            Self
        """
        self.is_fitted = True
        self.selected_features = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted selector

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if self.selector is not None:
            return self.selector.transform(X)
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Fit and transform features

        Args:
            X: Features to fit and transform
            y: Target array
            feature_names: List of feature names

        Returns:
            Transformed features
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available

        Returns:
            DataFrame with feature importance
        """
        return self.feature_importance

    def get_selected_features(self) -> Optional[list]:
        """
        Get list of selected feature names

        Returns:
            List of selected feature names
        """
        return self.selected_features