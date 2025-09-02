"""
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Preprocessor:
    """Handle feature preprocessing with awareness of feature types"""

    def __init__(self, config: Dict[str, Any], feature_types: Dict[str, str]):
        """
        Initialize Preprocessor

        Args:
            config: Configuration dictionary
            feature_types: Dictionary mapping feature names to types ("0" or "1")
        """
        self.config = config
        self.preprocessing_config = config['preprocessing']
        self.feature_types = feature_types

        self.preprocessor = None
        self.feature_names_out = None
        self.is_fitted = False

    def create_preprocessor(self, feature_names: list) -> ColumnTransformer:
        """
        Create preprocessing pipeline based on feature types

        Args:
            feature_names: List of feature names

        Returns:
            ColumnTransformer for preprocessing
        """
        method = self.preprocessing_config['method']

        if method == 'none':
            logger.info("No preprocessing applied")
            return None

        # Separate features by type
        additive_features = [col for col in feature_names
                           if self.feature_types.get(col, "0") == "0"]
        multiplicative_features = [col for col in feature_names
                                 if self.feature_types.get(col, "0") == "1"]

        logger.info(f"Additive features: {len(additive_features)}, "
                   f"Multiplicative features: {len(multiplicative_features)}")

        transformers = []

        if method == 'feature_aware':
            # Handle additive features
            if additive_features:
                scaler_type = self.preprocessing_config.get('scaler_additive', 'standard')
                additive_scaler = self._get_scaler(scaler_type)
                transformers.append(('additive', additive_scaler, additive_features))

            # Handle multiplicative features
            if multiplicative_features:
                mult_method = self.preprocessing_config.get('scaler_multiplicative', 'log')
                if mult_method == 'log':
                    # Use log transformation
                    mult_transformer = Pipeline([
                        ('log', LogTransformer()),
                        ('scale', StandardScaler())
                    ])
                elif mult_method == 'minmax':
                    mult_transformer = MinMaxScaler()
                else:
                    mult_transformer = 'passthrough'

                transformers.append(('multiplicative', mult_transformer, multiplicative_features))

        elif method == 'standard':
            # Apply same preprocessing to all features
            scaler = StandardScaler()
            transformers.append(('all', scaler, feature_names))

        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0,
            n_jobs=-1
        )

        return preprocessor

    def _get_scaler(self, scaler_type: str):
        """Get scaler based on type"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(scaler_type, StandardScaler())

    def fit(self, X: pd.DataFrame) -> 'Preprocessor':
        """
        Fit preprocessor on training data

        Args:
            X: Training features

        Returns:
            Self
        """
        logger.info("Fitting preprocessor...")

        self.preprocessor = self.create_preprocessor(list(X.columns))

        if self.preprocessor is not None:
            self.preprocessor.fit(X)
            # Get output feature names
            self.feature_names_out = self._get_feature_names_out(X.columns)
        else:
            self.feature_names_out = list(X.columns)

        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_out)}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features

        Args:
            X: Features to transform

        Returns:
            Transformed features array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X.values

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform features

        Args:
            X: Features to fit and transform

        Returns:
            Transformed features array
        """
        self.fit(X)
        return self.transform(X)

    def _get_feature_names_out(self, input_features: list) -> list:
        """Get feature names after transformation"""
        if self.preprocessor is None:
            return list(input_features)

        feature_names = []

        for name, transformer, features in self.preprocessor.transformers_:
            if transformer == 'passthrough':
                feature_names.extend(features)
            else:
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out(features)
                    feature_names.extend(names)
                else:
                    # For pipelines or transformers without get_feature_names_out
                    feature_names.extend([f"{name}_{f}" for f in features])

        return feature_names

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform features (if possible)

        Args:
            X: Transformed features

        Returns:
            Original scale features
        """
        if self.preprocessor is not None and hasattr(self.preprocessor, 'inverse_transform'):
            return self.preprocessor.inverse_transform(X)
        return X


class LogTransformer:
    """Custom transformer for log transformation"""

    def __init__(self, offset: float = 1.0):
        """
        Initialize LogTransformer

        Args:
            offset: Offset to add before log transformation
        """
        self.offset = offset

    def fit(self, X, y=None):
        """Fit transformer (no-op for log transform)"""
        return self

    def transform(self, X):
        """Apply log transformation"""
        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Apply log(1 + x) transformation to handle zero values
        # For multiplicative features (returns), this is appropriate
        return np.log1p(np.abs(X)) * np.sign(X)

    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """Inverse log transformation"""
        return np.sign(X) * (np.expm1(np.abs(X)))