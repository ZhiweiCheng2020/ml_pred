"""
Ensemble models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ensemble model"""
        super().__init__(config)
        self.model_name = 'ensemble'
        self.method = config.get('method', 'stacking')
        self.base_models = {}
        self.base_predictions = {}
        self.weights = None

    def build_model(self):
        """Build ensemble model"""
        if self.method == 'voting':
            return self._build_voting_ensemble()
        elif self.method == 'stacking':
            return self._build_stacking_ensemble()
        elif self.method == 'blending':
            return self._build_blending_ensemble()
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _build_voting_ensemble(self):
        """Build voting ensemble"""
        # This will be created dynamically with fitted base models
        return None

    def _build_stacking_ensemble(self):
        """Build stacking ensemble"""
        # Get meta model configuration
        stacking_config = self.config.get('stacking', {})
        meta_model_type = stacking_config.get('meta_model', 'ridge')
        meta_params = stacking_config.get('meta_model_params', {})

        if meta_model_type == 'ridge':
            meta_model = Ridge(**meta_params)
        elif meta_model_type == 'elasticnet':
            meta_model = ElasticNet(**meta_params)
        else:
            meta_model = Ridge(**meta_params)

        return meta_model

    def _build_blending_ensemble(self):
        """Build blending ensemble"""
        # Similar to stacking but with holdout validation
        blending_config = self.config.get('blending', {})
        meta_model_type = blending_config.get('meta_model', 'ridge')
        meta_params = blending_config.get('meta_model_params', {})

        if meta_model_type == 'ridge':
            meta_model = Ridge(**meta_params)
        else:
            meta_model = Ridge(**meta_params)

        return meta_model

    def set_base_models(self, base_models: Dict[str, BaseModel]):
        """
        Set base models for ensemble

        Args:
            base_models: Dictionary of fitted base models
        """
        self.base_models = base_models
        logger.info(f"Set {len(base_models)} base models for ensemble")

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'EnsembleModel':
        """Fit ensemble model"""
        logger.info(f"Training {self.model_name} model with {self.method} method...")

        if not self.base_models:
            raise ValueError("No base models set. Call set_base_models() first.")

        if self.method == 'voting':
            self._fit_voting(X, y, X_val, y_val)
        elif self.method == 'stacking':
            self._fit_stacking(X, y, X_val, y_val)
        elif self.method == 'blending':
            self._fit_blending(X, y, X_val, y_val)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def _fit_voting(self, X: np.ndarray, y: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None):
        """Fit voting ensemble"""
        # Get predictions from all base models
        predictions = []
        for name, model in self.base_models.items():
            if X_val is not None:
                pred = model.predict(X_val)
            else:
                pred = model.predict(X)
            predictions.append(pred)
            self.base_predictions[name] = pred

        predictions = np.array(predictions).T

        # Optimize weights if requested
        voting_config = self.config.get('voting', {})
        if voting_config.get('optimize_weights', True):
            self.weights = self._optimize_weights(predictions, y_val if y_val is not None else y)
        else:
            # Use uniform weights
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        logger.info(f"Voting weights: {dict(zip(self.base_models.keys(), self.weights))}")

    def _fit_stacking(self, X: np.ndarray, y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None):
        """Fit stacking ensemble"""
        stacking_config = self.config.get('stacking', {})
        use_oof = stacking_config.get('use_oof_predictions', True)
        cv_folds = stacking_config.get('cv_folds', 5)

        if use_oof:
            # Generate out-of-fold predictions
            meta_features = self._get_oof_predictions(X, y, cv_folds)
        else:
            # Use simple predictions
            meta_features = []
            for name, model in self.base_models.items():
                pred = model.predict(X)
                meta_features.append(pred)
            meta_features = np.column_stack(meta_features)

        # Train meta model
        self.model = self.build_model()
        self.model.fit(meta_features, y)

        # Store validation predictions if available
        if X_val is not None:
            val_meta_features = []
            for name, model in self.base_models.items():
                pred = model.predict(X_val)
                val_meta_features.append(pred)
                self.base_predictions[name] = pred
            val_meta_features = np.column_stack(val_meta_features)

    def _fit_blending(self, X: np.ndarray, y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None):
        """Fit blending ensemble"""
        blending_config = self.config.get('blending', {})
        blend_size = blending_config.get('blend_size', 0.2)

        # Split data for blending
        n_blend = int(len(X) * blend_size)
        X_blend, X_train = X[:n_blend], X[n_blend:]
        y_blend, y_train = y[:n_blend], y[n_blend:]

        # Get predictions on blend set
        blend_features = []
        for name, model in self.base_models.items():
            # Refit model on train subset
            model.fit(X_train, y_train)
            # Predict on blend set
            pred = model.predict(X_blend)
            blend_features.append(pred)

        blend_features = np.column_stack(blend_features)

        # Train meta model
        self.model = self.build_model()
        self.model.fit(blend_features, y_blend)

    def _get_oof_predictions(self, X: np.ndarray, y: np.ndarray, n_folds: int) -> np.ndarray:
        """Generate out-of-fold predictions"""
        n_samples = len(X)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"Generating OOF predictions for {name}")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train = y[train_idx]

                # Clone and refit model for this fold
                # Note: This is simplified - in practice you'd want to properly clone the model
                model.fit(X_fold_train, y_fold_train)

                # Predict on validation fold
                oof_predictions[val_idx, i] = model.predict(X_fold_val)

        return oof_predictions

    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights"""
        n_models = predictions.shape[1]

        # Objective function (MSE)
        def objective(weights):
            weighted_pred = np.sum(predictions * weights, axis=1)
            return np.mean((weighted_pred - y_true) ** 2)

        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # Initial weights (uniform)
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        return result.x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.method == 'voting':
            # Weighted average of predictions
            predictions = []
            for name, model in self.base_models.items():
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions).T
            return np.sum(predictions * self.weights, axis=1)

        elif self.method in ['stacking', 'blending']:
            # Get base model predictions
            meta_features = []
            for name, model in self.base_models.items():
                pred = model.predict(X)
                meta_features.append(pred)

            meta_features = np.column_stack(meta_features)

            # Meta model prediction
            return self.model.predict(meta_features)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")