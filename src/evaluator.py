"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


class Evaluator:
    """Handle model evaluation and metrics calculation"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Evaluator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.primary_metric = self.eval_config.get('primary_metric', 'rmse')
        self.secondary_metrics = self.eval_config.get('secondary_metrics', ['mae', 'r2'])

        # Multi-output support
        self.multi_output_config = self.eval_config.get('multi_output', {})
        self.is_multi_output = self.multi_output_config.get('enabled', False)
        self.n_outputs = self.multi_output_config.get('n_outputs', 1)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check if data is multi-output
        is_multi_output_data = len(y_true.shape) > 1 and y_true.shape[1] > 1

        if is_multi_output_data:
            return self._evaluate_multi_output(y_true, y_pred)
        else:
            return self._evaluate_single_output(y_true.flatten(), y_pred.flatten())

    def _evaluate_single_output(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate single output (original logic)"""
        metrics = {}

        # Primary metric
        metrics[self.primary_metric] = self._calculate_metric(
            y_true, y_pred, self.primary_metric
        )

        # Secondary metrics
        for metric_name in self.secondary_metrics:
            metrics[metric_name] = self._calculate_metric(
                y_true, y_pred, metric_name
            )

        # Always calculate MSE for compatibility
        if 'mse' not in metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)

        return metrics

    def _evaluate_multi_output(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """NEW: Evaluate multi-output"""
        metrics = {}
        n_outputs = y_true.shape[1]

        # Individual output metrics
        for i in range(n_outputs):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]

            # Calculate metrics for this output
            for metric_name in [self.primary_metric] + self.secondary_metrics:
                metric_value = self._calculate_metric(y_true_i, y_pred_i, metric_name)
                metrics[f"output_{i}_{metric_name}"] = metric_value

        # Overall metrics (average across outputs)
        for metric_name in [self.primary_metric] + self.secondary_metrics:
            if metric_name == 'rmse':
                overall_mse = np.mean((y_true - y_pred) ** 2)
                metrics[f"overall_{metric_name}"] = np.sqrt(overall_mse)
            elif metric_name == 'mse':
                metrics[f"overall_{metric_name}"] = np.mean((y_true - y_pred) ** 2)
            elif metric_name == 'mae':
                metrics[f"overall_{metric_name}"] = np.mean(np.abs(y_true - y_pred))
            elif metric_name == 'r2':
                # Use sklearn's multioutput scoring
                metrics[f"overall_{metric_name}"] = r2_score(y_true, y_pred, multioutput='uniform_average')
            else:
                # Average individual output metrics
                output_metrics = [metrics[f"output_{i}_{metric_name}"] for i in range(n_outputs)]
                metrics[f"overall_{metric_name}"] = np.mean(output_metrics)

        return metrics

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray,
                         metric_name: str) -> float:
        """
        Calculate a specific metric

        Args:
            y_true: True values
            y_pred: Predicted values
            metric_name: Name of the metric

        Returns:
            Metric value
        """
        if metric_name == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric_name == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric_name == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric_name == 'r2':
            return r2_score(y_true, y_pred)
        elif metric_name == 'mape':
            # Handle zero values in y_true
            mask = y_true != 0
            if np.sum(mask) == 0:
                return np.inf
            return mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            logger.warning(f"Unknown metric: {metric_name}. Using RMSE.")
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def evaluate_models(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comparison DataFrame from model results

        Args:
            results: Dictionary of model results

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue

            row = {
                'model': model_name,
                'training_time': model_results.get('training_time', 0),
                'n_features': model_results.get('n_features', 0)
            }

            # Add training metrics
            train_metrics = model_results.get('train_metrics', {})
            for metric_name, value in train_metrics.items():
                row[f'train_{metric_name}'] = value

            # Add validation metrics
            val_metrics = model_results.get('val_metrics', {})
            for metric_name, value in val_metrics.items():
                row[f'val_{metric_name}'] = value

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by overall metric if available, otherwise primary metric
        sort_candidates = [
            f'val_overall_{self.primary_metric}',
            f'val_{self.primary_metric}',
            f'train_overall_{self.primary_metric}',
            f'train_{self.primary_metric}'
        ]

        sort_column = None
        for candidate in sort_candidates:
            if candidate in comparison_df.columns:
                sort_column = candidate
                break

        if sort_column:
            ascending = self.primary_metric in ['rmse', 'mse', 'mae', 'mape']
            comparison_df = comparison_df.sort_values(sort_column, ascending=ascending)

        return comparison_df

    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate residual statistics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of residual statistics
        """
        residuals = y_true - y_pred

        stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q50': float(np.percentile(residuals, 50)),
            'q75': float(np.percentile(residuals, 75)),
            'skewness': float(self._calculate_skewness(residuals)),
            'kurtosis': float(self._calculate_kurtosis(residuals))
        }

        return stats

    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        n = len(x)
        if n < 3:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        return (n / ((n - 1) * (n - 2))) * np.sum(((x - mean) / std) ** 3)

    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis of distribution"""
        n = len(x)
        if n < 4:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mean) / std) ** 4) - \
               (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    def calculate_prediction_intervals(self, y_pred: np.ndarray,
                                      residuals: np.ndarray,
                                      confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals

        Args:
            y_pred: Predicted values
            residuals: Residuals from validation set
            confidence: Confidence level

        Returns:
            Dictionary with lower and upper bounds
        """
        # Calculate residual standard deviation
        residual_std = np.std(residuals)

        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

        # Calculate intervals
        margin = z_score * residual_std

        return {
            'lower': y_pred - margin,
            'upper': y_pred + margin,
            'margin': margin
        }

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of CV folds

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import cross_val_score, cross_validate

        # Define scoring metrics
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }

        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        # Process results
        results = {
            'cv_folds': cv_folds,
            'train_rmse_mean': -np.mean(cv_results['train_rmse']),
            'train_rmse_std': np.std(cv_results['train_rmse']),
            'test_rmse_mean': -np.mean(cv_results['test_rmse']),
            'test_rmse_std': np.std(cv_results['test_rmse']),
            'train_mae_mean': -np.mean(cv_results['train_mae']),
            'train_mae_std': np.std(cv_results['train_mae']),
            'test_mae_mean': -np.mean(cv_results['test_mae']),
            'test_mae_std': np.std(cv_results['test_mae']),
            'train_r2_mean': np.mean(cv_results['train_r2']),
            'train_r2_std': np.std(cv_results['train_r2']),
            'test_r2_mean': np.mean(cv_results['test_r2']),
            'test_r2_std': np.std(cv_results['test_r2'])
        }

        return results