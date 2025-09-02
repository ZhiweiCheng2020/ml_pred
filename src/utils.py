"""
Utility functions
"""

import yaml
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import datetime

logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', 'logs/pipeline.log')

    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Logging configured. Level: {log_level}, File: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")

    return config


def save_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    """
    Save training results

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison CSV
    if 'comparison' in results:
        comparison_path = output_dir / "model_comparison.csv"
        results['comparison'].to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")

    # Save detailed results as JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_{timestamp}.json"

    # Convert non-serializable objects
    serializable_results = make_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Detailed results saved to {results_path}")

    # Save individual model results
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_results in results.get('model_results', {}).items():
        model_path = models_dir / f"{model_name}_results.json"
        serializable_model_results = make_serializable(model_results)

        with open(model_path, 'w') as f:
            json.dump(serializable_model_results, f, indent=2)

        logger.info(f"{model_name} results saved to {model_path}")


def make_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to serializable format

    Args:
        obj: Object to convert

    Returns:
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create output directories based on configuration

    Args:
        config: Configuration dictionary
    """
    output_config = config.get('output', {})

    directories = [
        output_config.get('results_dir', 'results'),
        output_config.get('model_artifacts_dir', 'results/model_artifacts'),
        output_config.get('feature_importance_dir', 'results/feature_importance'),
        output_config.get('predictions_dir', 'results/predictions'),
        output_config.get('plots_dir', 'results/plots'),
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("Output directories created")


def save_predictions(predictions: np.ndarray,
                     y_true: Optional[np.ndarray],
                     model_name: str,
                     dataset_type: str,
                     output_dir: str = "results/predictions") -> None:
    """
    Save model predictions

    Args:
        predictions: Model predictions
        y_true: True values (optional)
        model_name: Name of the model
        dataset_type: Type of dataset (train/val/test)
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({'predictions': predictions})

    if y_true is not None:
        df['actual'] = y_true
        df['residual'] = y_true - predictions
        df['absolute_error'] = np.abs(df['residual'])
        df['percentage_error'] = np.where(
            df['actual'] != 0,
            100 * df['residual'] / df['actual'],
            np.nan
        )

    # Save to CSV
    filename = f"{model_name}_{dataset_type}_predictions.csv"
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)

    logger.info(f"Predictions saved to {filepath}")


def save_feature_importance(feature_importance: pd.DataFrame,
                            model_name: str,
                            output_dir: str = "results/feature_importance") -> None:
    """
    Save feature importance

    Args:
        feature_importance: Feature importance DataFrame
        model_name: Name of the model
        output_dir: Output directory
    """
    if feature_importance is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_feature_importance.csv"
    filepath = output_dir / filename

    feature_importance.to_csv(filepath, index=False)

    logger.info(f"Feature importance saved to {filepath}")


def load_model_results(results_path: str) -> Dict[str, Any]:
    """
    Load saved model results

    Args:
        results_path: Path to results file

    Returns:
        Results dictionary
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def format_metric_name(metric: str) -> str:
    """
    Format metric name for display

    Args:
        metric: Metric name

    Returns:
        Formatted metric name
    """
    formatting = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mse': 'MSE',
        'r2': 'R²',
        'mape': 'MAPE (%)'
    }

    return formatting.get(metric, metric.upper())


def calculate_improvement(baseline: float, current: float,
                          metric: str = 'rmse') -> float:
    """
    Calculate percentage improvement

    Args:
        baseline: Baseline metric value
        current: Current metric value
        metric: Metric name

    Returns:
        Percentage improvement
    """
    if baseline == 0:
        return 0

    if metric in ['rmse', 'mae', 'mse', 'mape']:
        # Lower is better
        improvement = (baseline - current) / baseline * 100
    else:
        # Higher is better
        improvement = (current - baseline) / abs(baseline) * 100

    return improvement