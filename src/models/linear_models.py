"""
Linear regression models with sample weight support
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge, ElasticNet
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RidgeModel(BaseModel):
    """Ridge Regression model with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ridge model"""
        super().__init__(config)
        self.model_name = 'ridge'

    def build_model(self) -> Ridge:
        """Build Ridge model with current parameters"""
        return Ridge(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'RidgeModel':
        """Fit Ridge model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build and fit model with sample weights
        self.model = self.build_model()

        # Ridge supports sample_weight parameter
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully with sample weights")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)


class ElasticNetModel(BaseModel):
    """ElasticNet Regression model with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ElasticNet model"""
        super().__init__(config)
        self.model_name = 'elasticnet'

    def build_model(self) -> ElasticNet:
        """Build ElasticNet model with current parameters"""
        return ElasticNet(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'ElasticNetModel':
        """Fit ElasticNet model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build and fit model with sample weights
        self.model = self.build_model()

        # ElasticNet supports sample_weight parameter
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully with sample weights")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)