"""
Model training orchestrator
"""

import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
import joblib
from typing import Dict, Any, Optional, Tuple, List
from .models import get_model
from .preprocessor import Preprocessor
from .feature_selector import FeatureSelector
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrate model training pipeline"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.results = {}
        self.preprocessors = {}
        self.feature_selectors = {}
        self.trained_models = {}

    def train_model(self,
                    model_name: str,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    weights_train: Optional[np.ndarray] = None,
                    weights_val: Optional[np.ndarray] = None,
                    feature_names: Optional[List[str]] = None,
                    feature_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Train a single model with sample weight support

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            weights_train: Training sample weights
            weights_val: Validation sample weights
            feature_names: List of feature names
            feature_types: Dictionary of feature types

        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training {model_name} model")
        logger.info(f"{'=' * 50}")

        # Log weight information if available
        if weights_train is not None:
            logger.info(f"Using sample weights - Train: mean={np.mean(weights_train):.3f}, "
                       f"high-weight samples: {np.sum(weights_train == 2.0)}/{len(weights_train)}")
        if weights_val is not None:
            logger.info(f"Using sample weights - Val: mean={np.mean(weights_val):.3f}, "
                       f"high-weight samples: {np.sum(weights_val == 2.0)}/{len(weights_val)}")

        # Load model configuration
        model_config_path = Path(f"config/models/{model_name}.yaml")
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        from .utils import load_config
        model_config = load_config(str(model_config_path))

        # Track timing
        start_time = time.time()

        # Preprocessing
        apply_preprocessing = model_name not in ['xgboost', 'lightgbm', 'catboost', 'random_forest']

        if apply_preprocessing and feature_types is not None:
            logger.info("Applying preprocessing...")
            preprocessor = Preprocessor(self.config, feature_types)
            X_train_processed = preprocessor.fit_transform(pd.DataFrame(X_train, columns=feature_names))
            X_val_processed = preprocessor.transform(
                pd.DataFrame(X_val, columns=feature_names)) if X_val is not None else None
            processed_feature_names = preprocessor.feature_names_out
            self.preprocessors[model_name] = preprocessor
        else:
            X_train_processed = X_train
            X_val_processed = X_val
            processed_feature_names = feature_names

        # Feature selection
        apply_selection = model_config.get('apply_feature_selection', True)

        if apply_selection:
            logger.info("Applying feature selection...")
            feature_selector = FeatureSelector(self.config)
            X_train_selected, selected_features = feature_selector.select_features(
                X_train_processed, y_train, processed_feature_names, model_name, apply_selection
            )

            if X_val_processed is not None:
                X_val_selected = feature_selector.transform(X_val_processed)
            else:
                X_val_selected = None

            self.feature_selectors[model_name] = feature_selector
            final_feature_names = selected_features
        else:
            X_train_selected = X_train_processed
            X_val_selected = X_val_processed
            final_feature_names = processed_feature_names

        logger.info(f"Final feature count: {X_train_selected.shape[1]}")

        # Initialize and train model
        model = get_model(model_name, model_config)
        model.fit(X_train_selected, y_train, X_val_selected, y_val)

        # Store trained model
        self.trained_models[model_name] = model

        # Calculate training time
        training_time = time.time() - start_time

        # Evaluate model
        evaluator = Evaluator(self.config)

        # Training predictions
        train_predictions = model.predict(X_train_selected)
        train_metrics = evaluator.evaluate(y_train, train_predictions, weights_train)

        # Validation predictions
        val_metrics = {}
        val_predictions = None
        if X_val_selected is not None and y_val is not None:
            val_predictions = model.predict(X_val_selected)
            val_metrics = evaluator.evaluate(y_val, val_predictions, weights_val)

        # Get feature importance if available
        feature_importance = model.get_feature_importance(final_feature_names)

        # Compile results
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'n_features': X_train_selected.shape[1],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'best_params': model.best_params,
            'train_predictions': train_predictions,
            'val_predictions': val_predictions,
            'used_sample_weights': weights_train is not None  # NEW FIELD
        }

        # Log results with primary metric
        logger.info(f"Training completed in {training_time:.2f} seconds")
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'rmse')

        if primary_metric in train_metrics:
            logger.info(f"Train {primary_metric.upper()}: {train_metrics[primary_metric]:.4f}")
        if val_metrics and primary_metric in val_metrics:
            logger.info(f"Val {primary_metric.upper()}: {val_metrics[primary_metric]:.4f}")

        # Log weight impact if weights were used
        if weights_train is not None:
            logger.info(f"Metrics calculated using sample weights (high-weight samples emphasized)")

        return results

    def train_all_models(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None,
                         weights_train: Optional[np.ndarray] = None,
                         weights_val: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None,
                         feature_types: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models specified in configuration with sample weight support

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            weights_train: Training sample weights
            weights_val: Validation sample weights
            feature_names: List of feature names
            feature_types: Dictionary of feature types

        Returns:
            Dictionary of results for all models
        """
        models_to_run = self.config.get('models_to_run', [])

        logger.info(f"Training {len(models_to_run)} models: {models_to_run}")

        if weights_train is not None:
            logger.info(f"Using weighted training with {np.sum(weights_train == 2.0)} high-weight samples")

        all_results = {}

        for model_name in models_to_run:
            if model_name == 'ensemble':
                # Skip ensemble for now, train it after base models
                continue

            try:
                results = self.train_model(
                    model_name, X_train, y_train, X_val, y_val,
                    weights_train, weights_val, feature_names, feature_types  # PASS WEIGHTS
                )
                all_results[model_name] = results
                self.results[model_name] = results

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[model_name] = {'error': str(e)}

        # Train ensemble if specified
        if 'ensemble' in models_to_run and len(self.trained_models) > 1:
            try:
                ensemble_results = self.train_ensemble(
                    X_train, y_train, X_val, y_val,
                    weights_train, weights_val, feature_names, feature_types  # PASS WEIGHTS
                )
                all_results['ensemble'] = ensemble_results
                self.results['ensemble'] = ensemble_results

            except Exception as e:
                logger.error(f"Error training ensemble: {str(e)}")
                all_results['ensemble'] = {'error': str(e)}

        # After training all models, save them for future predictions
        self.save_trained_models(
            self.config.get('output', {}).get('model_artifacts_dir', 'results/model_artifacts')
        )

        return all_results

    def get_best_model(self, metric: str = 'rmse', use_validation: bool = True) -> str:
        """
        Get the name of the best performing model

        Args:
            metric: Metric to use for comparison
            use_validation: Whether to use validation or training metrics

        Returns:
            Name of the best model
        """
        if not self.results:
            raise ValueError("No models trained yet")

        best_model = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mse'] else float('-inf')

        for model_name, results in self.results.items():
            if 'error' in results:
                continue

            if use_validation and results.get('val_metrics'):
                score = results['val_metrics'].get(metric, float('inf'))
            else:
                score = results['train_metrics'].get(metric, float('inf'))

            if metric in ['rmse', 'mae', 'mse']:
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model = model_name

        if best_model:
            logger.info(f"Best model: {best_model} with {metric}={best_score:.4f} (weighted)")  # NEW LOGGING

        return best_model

    def train_ensemble(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       weights_train: Optional[np.ndarray] = None,
                       weights_val: Optional[np.ndarray] = None,
                       feature_names: Optional[List[str]] = None,
                       feature_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Train ensemble model using trained base models with sample weight support

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            weights_train: Training sample weights
            weights_val: Validation sample weights
            feature_names: List of feature names
            feature_types: Dictionary of feature types

        Returns:
            Ensemble training results
        """
        logger.info(f"\n{'=' * 50}")
        logger.info("Training ensemble model")
        logger.info(f"{'=' * 50}")

        # Load ensemble configuration
        from .utils import load_config
        ensemble_config = load_config("config/models/ensemble.yaml")

        # Get base models to include
        base_model_names = ensemble_config.get('base_models', list(self.trained_models.keys()))
        base_model_names = [m for m in base_model_names if m in self.trained_models]

        if len(base_model_names) < 2:
            raise ValueError(f"Need at least 2 base models for ensemble, got {len(base_model_names)}")

        logger.info(f"Using base models: {base_model_names}")

        # Initialize ensemble model
        from .models import EnsembleModel
        ensemble_model = EnsembleModel(ensemble_config)

        # Prepare base models with their preprocessing
        processed_base_models = {}
        for model_name in base_model_names:
            processed_base_models[model_name] = self.trained_models[model_name]

        ensemble_model.set_base_models(processed_base_models)

        # Train ensemble
        start_time = time.time()
        ensemble_model.fit(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # Evaluate ensemble
        evaluator = Evaluator(self.config)

        # Training predictions
        train_predictions = ensemble_model.predict(X_train)
        train_metrics = evaluator.evaluate(y_train, train_predictions, weights_train)

        # Validation predictions
        val_metrics = {}
        val_predictions = None
        if X_val is not None and y_val is not None:
            val_predictions = ensemble_model.predict(X_val)
            val_metrics = evaluator.evaluate(y_val, val_predictions, weights_val)

        # Store ensemble model
        self.trained_models['ensemble'] = ensemble_model

        # Compile results
        results = {
            'model_name': 'ensemble',
            'ensemble_method': ensemble_config.get('method', 'stacking'),
            'base_models': base_model_names,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_predictions': train_predictions,
            'val_predictions': val_predictions,
            'used_sample_weights': weights_train is not None
        }

        # Log results
        logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'rmse')

        if primary_metric in train_metrics:
            logger.info(f"Train {primary_metric.upper()}: {train_metrics[primary_metric]:.4f}")
        if val_metrics and primary_metric in val_metrics:
            logger.info(f"Val {primary_metric.upper()}: {val_metrics[primary_metric]:.4f}")

        return results

    def save_trained_models(self, output_dir: str = "results/model_artifacts") -> None:
        """
        Save all trained models and their components for future predictions

        Args:
            output_dir: Directory to save model artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving trained models to {output_dir}")

        for model_name, model in self.trained_models.items():
            try:
                # Save the model
                model_path = output_path / f"{model_name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} model to {model_path}")

                # Save preprocessor if exists
                if model_name in self.preprocessors:
                    preprocessor_path = output_path / f"{model_name}_preprocessor.pkl"
                    joblib.dump(self.preprocessors[model_name], preprocessor_path)
                    logger.info(f"Saved {model_name} preprocessor")

                # Save feature selector if exists
                if model_name in self.feature_selectors:
                    selector_path = output_path / f"{model_name}_feature_selector.pkl"
                    joblib.dump(self.feature_selectors[model_name], selector_path)
                    logger.info(f"Saved {model_name} feature selector")

            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")