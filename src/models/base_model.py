"""
Base model class with sample weight support
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import optuna

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models with sample weight support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', 'base_model')
        self.model_type = config.get('model_type', 'unknown')
        self.parameters = config.get('parameters', {})
        self.hyperparameter_tuning = config.get('hyperparameter_tuning', {})

        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.is_fitted = False

    @abstractmethod
    def build_model(self) -> Any:
        """Build the model with given parameters"""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'BaseModel':
        """
        Fit the model with sample weight support

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Training sample weights (optional)
            val_sample_weight: Validation sample weights (optional)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        pass

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None,
                            sample_weight: Optional[np.ndarray] = None,
                            val_sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters using specified method with sample weight support

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Training sample weights (optional)
            val_sample_weight: Validation sample weights (optional)

        Returns:
            Best parameters
        """
        if not self.hyperparameter_tuning.get('enabled', False):
            logger.info(f"Hyperparameter tuning disabled for {self.model_name}")
            return self.parameters

        method = self.hyperparameter_tuning.get('method', 'grid_search')
        logger.info(f"Starting {method} hyperparameter tuning for {self.model_name}")

        if method == 'grid_search':
            best_params = self._grid_search(X, y, sample_weight)
        elif method == 'random_search':
            best_params = self._random_search(X, y, sample_weight)
        elif method == 'optuna':
            best_params = self._optuna_search(X, y, X_val, y_val, sample_weight, val_sample_weight)
        else:
            logger.warning(f"Unknown tuning method: {method}. Using default parameters.")
            best_params = self.parameters

        self.best_params = best_params
        logger.info(f"Best parameters for {self.model_name}: {best_params}")

        return best_params

    def _grid_search(self, X: np.ndarray, y: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform grid search with sample weight support"""
        param_grid = self.hyperparameter_tuning.get('param_grid', {})
        cv_folds = self.hyperparameter_tuning.get('cv_folds', 5)
        scoring = self.hyperparameter_tuning.get('scoring', 'neg_mean_squared_error')

        base_model = self.build_model()

        # Pass sample_weight to fit_params if the model supports it
        fit_params = {}
        if sample_weight is not None and hasattr(base_model, 'fit') and 'sample_weight' in base_model.fit.__code__.co_varnames:
            fit_params['sample_weight'] = sample_weight

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        if fit_params:
            grid_search.fit(X, y, **fit_params)
        else:
            grid_search.fit(X, y)

        return grid_search.best_params_

    def _random_search(self, X: np.ndarray, y: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform random search with sample weight support"""
        param_distributions = self.hyperparameter_tuning.get('param_distributions', {})
        n_iter = self.hyperparameter_tuning.get('n_iter', 20)
        cv_folds = self.hyperparameter_tuning.get('cv_folds', 5)
        scoring = self.hyperparameter_tuning.get('scoring', 'neg_mean_squared_error')

        base_model = self.build_model()

        fit_params = {}
        if sample_weight is not None and hasattr(base_model, 'fit') and 'sample_weight' in base_model.fit.__code__.co_varnames:
            fit_params['sample_weight'] = sample_weight

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        if fit_params:
            random_search.fit(X, y, **fit_params)
        else:
            random_search.fit(X, y)

        return random_search.best_params_

    def _optuna_search(self, X: np.ndarray, y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None,
                      val_sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform Optuna optimization with sample weight support"""
        n_trials = self.hyperparameter_tuning.get('n_trials', 50)
        param_space = self.hyperparameter_tuning.get('param_space', {})

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list) and len(param_range) == 2:
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range[0], float) or isinstance(param_range[1], float):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)

            # Update model parameters
            self.parameters.update(params)
            model = self.build_model()

            # Evaluate using validation set or cross-validation
            if X_val is not None and y_val is not None:
                # Use validation set
                try:
                    if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                        model.fit(X, y, sample_weight=sample_weight)
                    else:
                        model.fit(X, y)
                    y_pred = model.predict(X_val)

                    # Calculate weighted MSE for validation
                    if val_sample_weight is not None:
                        mse = np.average((y_val - y_pred) ** 2, weights=val_sample_weight)
                    else:
                        mse = np.mean((y_val - y_pred) ** 2)
                    score = -mse  # Negative MSE for maximization
                except:
                    score = float('-inf')
            else:
                # Use cross-validation
                cv_folds = self.hyperparameter_tuning.get('cv_folds', 5)
                try:
                    if sample_weight is not None:
                        # For CV with weights, we'll use a simple approach
                        # More sophisticated weighted CV could be implemented if needed
                        scores = cross_val_score(model, X, y, cv=cv_folds,
                                               scoring='neg_mean_squared_error', n_jobs=-1)
                    else:
                        scores = cross_val_score(model, X, y, cv=cv_folds,
                                               scoring='neg_mean_squared_error', n_jobs=-1)
                    score = scores.mean()
                except:
                    score = float('-inf')

            return score

        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available

        Args:
            feature_names: Optional list of feature names

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. No feature importance available.")
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
            if len(importances.shape) > 1:
                importances = importances.flatten()
        else:
            return None

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        if len(feature_names) != len(importances):
            logger.warning(f"Feature names length ({len(feature_names)}) doesn't match "
                         f"importances length ({len(importances)})")
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return self.feature_importance

    def save_model(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Nothing to save.")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'parameters': self.parameters,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> 'BaseModel':
        """
        Load model from disk

        Args:
            path: Path to load model from

        Returns:
            Self
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.parameters = model_data['parameters']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = True

        logger.info(f"Model loaded from {path}")

        return self