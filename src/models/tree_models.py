"""
Tree-based models
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest Regression model"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model"""
        super().__init__(config)
        self.model_name = 'random_forest'

    def build_model(self) -> RandomForestRegressor:
        """Build Random Forest model with current parameters"""
        return RandomForestRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'RandomForestModel':
        """Fit Random Forest model"""
        logger.info(f"Training {self.model_name} model...")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val)
            self.parameters.update(best_params)

        # Build and fit model
        self.model = self.build_model()
        self.model.fit(X, y)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost Regression model"""

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

        return xgb.XGBRegressor(**model_params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'XGBoostModel':
        """Fit XGBoost model"""
        logger.info(f"Training {self.model_name} model...")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val)
            self.parameters.update(best_params)

        # Build model
        self.model = self.build_model()

        # Prepare fit parameters
        fit_params = {}

        # Add eval_set if validation data is provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

            # Add early stopping if enabled in config
            early_stopping_config = self.config.get('early_stopping', {})
            if early_stopping_config.get('enabled', False):
                fit_params['early_stopping_rounds'] = early_stopping_config.get('rounds', 50)

        # Fit model
        self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM Regression model"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model"""
        super().__init__(config)
        self.model_name = 'lightgbm'

    def build_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model with current parameters"""
        return lgb.LGBMRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'LightGBMModel':
        """Fit LightGBM model"""
        logger.info(f"Training {self.model_name} model...")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val)
            self.parameters.update(best_params)

        # Build model
        self.model = self.build_model()

        # Prepare eval set for early stopping
        eval_set = None
        callbacks = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if self.config.get('early_stopping', {}).get('enabled', False):
                early_stopping_rounds = self.config['early_stopping'].get('rounds', 50)
                callbacks = [lgb.early_stopping(early_stopping_rounds),
                             lgb.log_evaluation(period=0)]

        # Fit model
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if callbacks is not None:
            fit_params['callbacks'] = callbacks

        self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)


class CatBoostModel(BaseModel):
    """CatBoost Regression model"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize CatBoost model"""
        super().__init__(config)
        self.model_name = 'catboost'

    def build_model(self) -> cb.CatBoostRegressor:
        """Build CatBoost model with current parameters"""
        return cb.CatBoostRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'CatBoostModel':
        """Fit CatBoost model"""
        logger.info(f"Training {self.model_name} model...")

        # Tune hyperparameters if enabled
        if self.hyperparameter_tuning.get('enabled', False):
            best_params = self.tune_hyperparameters(X, y, X_val, y_val)
            self.parameters.update(best_params)

        # Build model
        self.model = self.build_model()

        # Prepare eval set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = cb.Pool(X_val, y_val)

        # Fit model
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['use_best_model'] = True

        self.model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)