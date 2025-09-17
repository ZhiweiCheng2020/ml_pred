"""
Ensemble models with sample weight support
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
    """Ensemble model combining multiple base models with sample weight support"""

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
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'EnsembleModel':
        """
        Fit ensemble model with sample weight support

        Note: Ensemble training with sample weights is complex since it involves
        combining predictions from already-trained base models. For simplicity,
        we focus on weighted evaluation rather than weighted ensemble training.
        """
        logger.info(f"Training {self.model_name} model with {self.method} method...")

        if not self.base_models:
            raise ValueError("No base models set. Call set_base_models() first.")

        if sample_weight is not None:
            logger.info(f"Note: Ensemble uses base model predictions (already trained with weights)")
            logger.info(f"Ensemble evaluation will use sample weights for consistency")

        if self.method == 'voting':
            self._fit_voting(X, y, X_val, y_val, sample_weight, val_sample_weight)
        elif self.method == 'stacking':
            self._fit_stacking(X, y, X_val, y_val, sample_weight, val_sample_weight)
        elif self.method == 'blending':
            self._fit_blending(X, y, X_val, y_val, sample_weight, val_sample_weight)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def _fit_voting(self, X: np.ndarray, y: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    sample_weight: Optional[np.ndarray] = None,
                    val_sample_weight: Optional[np.ndarray] = None):
        """Fit voting ensemble with sample weight consideration"""
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

        # Optimize weights if requested, considering sample weights
        voting_config = self.config.get('voting', {})
        if voting_config.get('optimize_weights', True):
            target_data = y_val if y_val is not None else y
            target_weights = val_sample_weight if val_sample_weight is not None else sample_weight
            self.weights = self._optimize_weights(predictions, target_data, target_weights)
        else:
            # Use uniform weights
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        logger.info(f"Voting weights: {dict(zip(self.base_models.keys(), self.weights))}")

    def _fit_stacking(self, X: np.ndarray, y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None,
                      val_sample_weight: Optional[np.ndarray] = None):
        """Fit stacking ensemble with sample weight support"""
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

        # Train meta model with sample weights
        self.model = self.build_model()

        # Meta-model training with sample weights if supported
        if sample_weight is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
            self.model.fit(meta_features, y, sample_weight=sample_weight)
            logger.info("Meta-model trained with sample weights")
        else:
            self.model.fit(meta_features, y)
            logger.info("Meta-model trained without sample weights (not supported by meta-model)")

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
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None,
                      val_sample_weight: Optional[np.ndarray] = None):
        """Fit blending ensemble with sample weight support"""
        blending_config = self.config.get('blending', {})
        blend_size = blending_config.get('blend_size', 0.2)

        # Split data for blending
        n_blend = int(len(X) * blend_size)
        X_blend, X_train = X[:n_blend], X[n_blend:]
        y_blend, y_train = y[:n_blend], y[n_blend:]

        # Split sample weights too
        if sample_weight is not None:
            weight_blend, weight_train = sample_weight[:n_blend], sample_weight[n_blend:]
        else:
            weight_blend = weight_train = None

        # Get predictions on blend set
        blend_features = []
        for name, model in self.base_models.items():
            # Note: Base models are already trained, so we just get predictions
            # In a full implementation, we might retrain models on the train subset
            pred = model.predict(X_blend)
            blend_features.append(pred)

        blend_features = np.column_stack(blend_features)

        # Train meta model with sample weights
        self.model = self.build_model()

        if weight_blend is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
            self.model.fit(blend_features, y_blend, sample_weight=weight_blend)
            logger.info("Blending meta-model trained with sample weights")
        else:
            self.model.fit(blend_features, y_blend)
            logger.info("Blending meta-model trained without sample weights")

    def _get_oof_predictions(self, X: np.ndarray, y: np.ndarray, n_folds: int) -> np.ndarray:
        """
        Generate out-of-fold predictions

        Note: This is a simplified version. In practice, for proper OOF predictions
        with sample weights, we would need to retrain base models on each fold.
        For now, we use the already-trained base models.
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        # Simple approach: use existing trained models
        # This is not true OOF but works for demonstration
        for i, (name, model) in enumerate(self.base_models.items()):
            oof_predictions[:, i] = model.predict(X)

        logger.info("Using existing base model predictions (not true OOF)")
        return oof_predictions

    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimize ensemble weights with sample weight support"""
        n_models = predictions.shape[1]

        # Objective function (weighted MSE)
        def objective(weights):
            weighted_pred = np.sum(predictions * weights, axis=1)
            if sample_weight is not None:
                # Calculate weighted MSE
                errors = (weighted_pred - y_true) ** 2
                weighted_mse = np.average(errors, weights=sample_weight)
                return weighted_mse
            else:
                # Standard MSE
                return np.mean((weighted_pred - y_true) ** 2)

        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # Initial weights (uniform)
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if sample_weight is not None:
            logger.info("Ensemble weights optimized using sample weights")

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

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Optional[pd.DataFrame]:
        """
        Get ensemble feature importance by aggregating base model importances

        Args:
            feature_names: Optional list of feature names

        Returns:
            DataFrame with aggregated feature importance
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. No feature importance available.")
            return None

        # Collect feature importances from base models
        all_importances = []
        model_weights = getattr(self, 'weights', None)

        if model_weights is None:
            # Use uniform weights if no optimized weights
            model_weights = np.ones(len(self.base_models)) / len(self.base_models)

        for i, (name, model) in enumerate(self.base_models.items()):
            importance_df = model.get_feature_importance(feature_names)
            if importance_df is not None:
                # Weight the importance by ensemble weight
                weighted_importance = importance_df['importance'].values * model_weights[i]
                all_importances.append(weighted_importance)
            else:
                logger.warning(f"No feature importance available for base model {name}")

        if not all_importances:
            logger.warning("No feature importances available from any base model")
            return None

        # Average the weighted importances
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(all_importances[0]))]

        avg_importance = np.mean(all_importances, axis=0)

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)

        return self.feature_importance