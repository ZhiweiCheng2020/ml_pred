"""
Tree-based models with sample weight support and multi-output compatibility
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest Regression model with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model"""
        super().__init__(config)
        self.model_name = 'random_forest'

    def build_model(self) -> RandomForestRegressor:
        """Build Random Forest model with current parameters"""
        return RandomForestRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'RandomForestModel':
        """Fit Random Forest model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build and fit model
        self.model = self.build_model()

        # RandomForest supports sample_weight parameter
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


class XGBoostModel(BaseModel):
    """XGBoost Regression model with sample weight support and native multi-output"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model"""
        super().__init__(config)
        self.model_name = 'xgboost'

    def build_model(self) -> xgb.XGBRegressor:
        """Build XGBoost model with current parameters"""
        # Remove early stopping parameters from model parameters as they should be passed to fit()
        model_params = self.parameters.copy()

        # These parameters should not be in the model constructor
        fit_only_params = ['early_stopping_rounds', 'eval_metric']
        for param in fit_only_params:
            model_params.pop(param, None)

        # Add multi-output support parameters if not present
        if 'tree_method' not in model_params:
            model_params['tree_method'] = 'hist'
        if 'multi_strategy' not in model_params:
            model_params['multi_strategy'] = 'multi_output_tree'

        return xgb.XGBRegressor(**model_params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'XGBoostModel':
        """Fit XGBoost model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        # Check if multi-output
        is_multioutput = len(y.shape) > 1 and y.shape[1] > 1
        if is_multioutput:
            logger.info(f"Detected multi-output data: {y.shape[1]} outputs")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build model
        self.model = self.build_model()

        # Prepare fit parameters
        fit_params = {}

        # Add sample weights - XGBoost supports sample_weight
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        # Add eval_set if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

            # Add early stopping if enabled in config
            early_stopping_config = self.config.get('early_stopping', {})
            if early_stopping_config.get('enabled', False):
                fit_params['early_stopping_rounds'] = early_stopping_config.get('rounds', 50)

        # Fit model with sample weights
        self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully with sample weights")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM Regression model with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model"""
        super().__init__(config)
        self.model_name = 'lightgbm'
        self.is_multioutput = False
        self.base_model = None

    def build_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model with current parameters"""
        return lgb.LGBMRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'LightGBMModel':
        """Fit LightGBM model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        # Check if multi-output
        self.is_multioutput = len(y.shape) > 1 and y.shape[1] > 1

        if self.is_multioutput:
            logger.info(f"Using MultiOutputRegressor for {y.shape[1]} outputs")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build base model
        self.base_model = self.build_model()

        # Wrap with MultiOutputRegressor if multi-output
        if self.is_multioutput:
            self.model = MultiOutputRegressor(self.base_model)
        else:
            self.model = self.base_model

        # For multi-output, we can't use LightGBM's native eval_set with MultiOutputRegressor
        # So we fit without early stopping for multi-output case
        if self.is_multioutput:
            # Simple fit for multi-output with sample weights
            if sample_weight is not None:
                self.model.fit(X, y, sample_weight=sample_weight)
            else:
                self.model.fit(X, y)
        else:
            # Original single-output logic with eval_set and sample weights
            eval_set = None
            callbacks = None

            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                if self.config.get('early_stopping', {}).get('enabled', False):
                    early_stopping_rounds = self.config['early_stopping'].get('rounds', 50)
                    callbacks = [lgb.early_stopping(early_stopping_rounds),
                                 lgb.log_evaluation(period=0)]

            # Fit model with sample weights
            fit_params = {}
            if eval_set is not None:
                fit_params['eval_set'] = eval_set
            if callbacks is not None:
                fit_params['callbacks'] = callbacks
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight

            self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully with sample weights")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Optional['pd.DataFrame']:
        """Get feature importance - handle both single and multi-output cases"""
        if not self.is_fitted:
            logger.warning("Model not fitted. No feature importance available.")
            return None

        if self.is_multioutput:
            # For MultiOutputRegressor, we need to aggregate feature importance across estimators
            try:
                import pandas as pd

                # Get feature importance from each estimator
                importances_list = []
                for i, estimator in enumerate(self.model.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        importances_list.append(estimator.feature_importances_)

                if importances_list:
                    # Average importance across all estimators
                    avg_importances = np.mean(importances_list, axis=0)

                    if feature_names is None:
                        feature_names = [f'feature_{i}' for i in range(len(avg_importances))]

                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': avg_importances
                    }).sort_values('importance', ascending=False)

                    self.feature_importance = feature_importance_df
                    return feature_importance_df
            except Exception as e:
                logger.warning(f"Could not extract feature importance for multi-output model: {e}")
                return None
        else:
            # Single output - use parent method
            return super().get_feature_importance(feature_names)

        return None


class CatBoostModel(BaseModel):
    """CatBoost Regression model with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize CatBoost model"""
        super().__init__(config)
        self.model_name = 'catboost'

    def build_model(self) -> cb.CatBoostRegressor:
        """Build CatBoost model with current parameters"""
        return cb.CatBoostRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'CatBoostModel':
        """Fit CatBoost model with sample weight support"""
        logger.info(f"Training {self.model_name} model...")

        if sample_weight is not None:
            logger.info(f"Using sample weights in training: mean={np.mean(sample_weight):.3f}, "
                       f"high-weight samples: {np.sum(sample_weight == 2.0)}/{len(sample_weight)}")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val, sample_weight, val_sample_weight)
            self.parameters.update(best_params)

        # Build model
        self.model = self.build_model()

        # Prepare eval set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            if val_sample_weight is not None:
                eval_set = cb.Pool(X_val, y_val, weight=val_sample_weight)
            else:
                eval_set = cb.Pool(X_val, y_val)

        # Prepare training data with sample weights
        if sample_weight is not None:
            train_pool = cb.Pool(X, y, weight=sample_weight)
        else:
            train_pool = cb.Pool(X, y)

        # Fit model with sample weights
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['use_best_model'] = True

        if sample_weight is not None:
            # Use Pool for training with weights
            self.model.fit(train_pool, **fit_params)
        else:
            self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully with sample weights")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)