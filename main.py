"""
Main pipeline execution script
"""

import argparse
import logging
from pathlib import Path
import sys
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_selector import FeatureSelector
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualization import Visualizer
from src.utils import (
    setup_logging, load_config, save_results, create_output_directories,
    save_predictions, save_feature_importance
)

logger = logging.getLogger(__name__)


def main(config_path: str = "config/config.yaml"):
    """
    Main pipeline execution

    Args:
        config_path: Path to main configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("Starting ML Pipeline")
    logger.info("=" * 60)

    # Create output directories
    create_output_directories(config)

    # Initialize components
    data_loader = DataLoader(config)
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    trainer = Trainer(config)

    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    X, y, feature_types = data_loader.load_data()

    # Print data summary
    data_summary = data_loader.get_data_summary()
    logger.info(f"Dataset summary:")
    for key, value in data_summary.items():
        logger.info(f"  {key}: {value}")

    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

    # Further split training data for validation
    val_size = 0.2
    val_samples = int(len(X_train) * val_size)

    X_val = X_train[:val_samples]
    y_val = y_train[:val_samples]
    X_train_final = X_train[val_samples:]
    y_train_final = y_train[val_samples:]

    logger.info(f"Final split - Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)

    # Convert to numpy arrays for training
    X_train_np = X_train_final.values if hasattr(X_train_final, 'values') else X_train_final
    y_train_np = y_train_final.values if hasattr(y_train_final, 'values') else y_train_final
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Get feature names
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]

    # Train all models
    all_results = trainer.train_all_models(
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        feature_names, feature_types
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    test_results = {}
    for model_name, model_results in all_results.items():
        if 'error' in model_results:
            logger.warning(f"Skipping {model_name} due to training error: {model_results['error']}")
            continue

        try:
            # Get the trained model
            model = trainer.trained_models[model_name]

            # Apply same preprocessing as training
            X_test_processed = X_test_np
            if model_name in trainer.preprocessors:
                preprocessor = trainer.preprocessors[model_name]
                X_test_processed = preprocessor.transform(pd.DataFrame(X_test_np, columns=feature_names))

            if model_name in trainer.feature_selectors:
                feature_selector = trainer.feature_selectors[model_name]
                X_test_processed = feature_selector.transform(X_test_processed)

            # Make predictions
            test_predictions = model.predict(X_test_processed)
            test_metrics = evaluator.evaluate(y_test_np, test_predictions)

            test_results[model_name] = {
                'test_metrics': test_metrics,
                'test_predictions': test_predictions
            }

            # Update results
            all_results[model_name]['test_metrics'] = test_metrics
            all_results[model_name]['test_predictions'] = test_predictions

            logger.info(f"{model_name} - Test RMSE: {test_metrics.get('rmse', 0):.4f}")

            # Save predictions
            save_predictions(
                test_predictions, y_test_np, model_name, 'test',
                config.get('output', {}).get('predictions_dir', 'results/predictions')
            )

            # Save feature importance
            if model_results.get('feature_importance') is not None:
                save_feature_importance(
                    model_results['feature_importance'], model_name,
                    config.get('output', {}).get('feature_importance_dir', 'results/feature_importance')
                )

        except Exception as e:
            logger.error(f"Error evaluating {model_name} on test set: {str(e)}")

    # Create model comparison
    logger.info("\n" + "=" * 60)
    logger.info("CREATING COMPARISON")
    logger.info("=" * 60)

    comparison_df = evaluator.evaluate_models(all_results)
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))

    # Save comparison
    comparison_path = config.get('output', {}).get('comparison_file', 'results/model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")

    # Visualizations
    logger.info("\n" + "=" * 60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 60)

    # Model comparison plot
    visualizer.plot_model_comparison(comparison_df, 'rmse')

    # Best model analysis
    best_model_name = trainer.get_best_model('rmse', use_validation=True)
    logger.info(f"Best model: {best_model_name}")

    if best_model_name and best_model_name in test_results:
        best_test_results = test_results[best_model_name]

        # Predictions vs actual plot
        visualizer.plot_predictions_vs_actual(
            y_test_np, best_test_results['test_predictions'],
            best_model_name, 'test'
        )

        # Residual distribution
        visualizer.plot_residual_distribution(
            y_test_np, best_test_results['test_predictions'],
            best_model_name
        )

        # Feature importance for best model
        if all_results[best_model_name].get('feature_importance') is not None:
            visualizer.plot_feature_importance(
                all_results[best_model_name]['feature_importance'],
                best_model_name
            )

    # Save all results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    final_results = {
        'comparison': comparison_df,
        'model_results': all_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'config': config
    }

    save_results(final_results, config.get('output', {}).get('results_dir', 'results'))

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.config)