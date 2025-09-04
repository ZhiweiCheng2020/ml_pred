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
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for log transformation with sklearn compatibility"""

    def __init__(self, offset: float = 1.0):
        """
        Initialize LogTransformer

        Args:
            offset: Offset to add before log transformation
        """
        self.offset = offset
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """Fit transformer (no-op for log transform)"""
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(n_features)])
        return self

    def transform(self, X):
        """Apply log transformation"""
        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Apply log(1 + x) transformation to handle zero values
        # For multiplicative features (returns), this is appropriate
        return np.log1p(np.abs(X_array)) * np.sign(X_array)

    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """Inverse log transformation"""
        return np.sign(X) * (np.expm1(np.abs(X)))

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation"""
        if input_features is not None:
            return np.array([f"log_{name}" for name in input_features])
        elif self.feature_names_in_ is not None:
            return np.array([f"log_{name}" for name in self.feature_names_in_])
        else:
            # If no feature names available, return generic names
            n_features = getattr(self, 'n_features_in_', 1)
            return np.array([f"log_feature_{i}" for i in range(n_features)])


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
            n_jobs=-1,
            verbose_feature_names_out=False  # Use shorter feature names
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

        # Try to use sklearn's get_feature_names_out if available
        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                # This works for sklearn >= 1.0
                feature_names = list(self.preprocessor.get_feature_names_out())
            else:
                # Fallback for older sklearn versions
                feature_names = self._get_feature_names_manual(input_features)
        except Exception as e:
            logger.warning(f"Could not get feature names automatically: {e}")
            # Fallback: create generic names
            feature_names = self._get_feature_names_manual(input_features)

        return feature_names

    def _get_feature_names_manual(self, input_features: list) -> list:
        """Manually construct feature names when automatic extraction fails"""
        if self.preprocessor is None:
            return list(input_features)

        feature_names = []

        # Get the transformers
        if hasattr(self.preprocessor, 'transformers_'):
            for name, transformer, features in self.preprocessor.transformers_:
                if transformer == 'passthrough':
                    feature_names.extend(features)
                elif transformer == 'drop':
                    continue
                else:
                    # Handle different transformer types
                    if isinstance(transformer, Pipeline):
                        # For pipelines, use the step names
                        prefix = name
                        for feature in features:
                            feature_names.append(f"{prefix}_{feature}")
                    elif hasattr(transformer, 'get_feature_names_out'):
                        try:
                            names = transformer.get_feature_names_out(features)
                            feature_names.extend(names)
                        except:
                            # If get_feature_names_out fails, use generic names
                            for feature in features:
                                feature_names.append(f"{name}_{feature}")
                    else:
                        # For transformers without get_feature_names_out
                        for feature in features:
                            feature_names.append(f"{name}_{feature}")

        # Handle remainder columns
        if hasattr(self.preprocessor, 'remainder') and self.preprocessor.remainder == 'passthrough':
            # Add any remaining features that weren't explicitly transformed
            all_transformed = set()
            for _, _, features in self.preprocessor.transformers_:
                if features != 'drop':
                    all_transformed.update(features)

            remainder_features = [f for f in input_features if f not in all_transformed]
            feature_names.extend(remainder_features)

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