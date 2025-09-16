"""
Main pipeline execution script with comprehensive visualization for all models
"""

import argparse
import logging
from pathlib import Path
import sys
import warnings
import pandas as pd
import numpy as np
import traceback

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


def visualize_model_results(model_name: str,
                           model_results: dict,
                           test_results: dict,
                           y_test: pd.Series,
                           visualizer: Visualizer,
                           config: dict) -> None:
    """
    Create visualizations for a single model

    Args:
        model_name: Name of the model
        model_results: Training and validation results
        test_results: Test set results
        y_test: True test values
        visualizer: Visualizer instance
        config: Configuration dictionary
    """
    logger.info(f"Creating visualizations for {model_name}...")

    try:
        # Get test predictions if available
        if model_name in test_results and 'test_predictions' in test_results[model_name]:
            test_predictions = test_results[model_name]['test_predictions']

            # 1. Predictions vs Actual plot
            visualizer.plot_predictions_vs_actual(
                y_test, test_predictions,
                model_name, 'test'
            )

            # 2. Residual distribution plot
            visualizer.plot_residual_distribution(
                y_test, test_predictions,
                model_name
            )

        # 3. Feature importance plot (if available)
        if model_results.get('feature_importance') is not None:
            visualizer.plot_feature_importance(
                model_results['feature_importance'],
                model_name,
                top_n=30  # Show top 30 features
            )

        # 4. Learning curves (if available)
        if 'learning_history' in model_results:
            history = model_results['learning_history']
            if 'train_scores' in history and 'val_scores' in history:
                visualizer.plot_learning_curves(
                    history['train_scores'],
                    history['val_scores'],
                    model_name,
                    metric=config.get('evaluation', {}).get('primary_metric', 'rmse')
                )

    except Exception as e:
        logger.error(f"Error creating visualizations for {model_name}: {str(e)}")
        logger.error(traceback.format_exc())


def main(config_path: str = "config/config.yaml"):
    """
    Main pipeline execution with sample weight support

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

    try:
        X, y, feature_types = data_loader.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    # Print data summary
    data_summary = data_loader.get_data_summary()
    logger.info(f"Dataset summary:")
    for key, value in data_summary.items():
        logger.info(f"  {key}: {value}")

    # Split data
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING DATA WITH SAMPLE WEIGHTS")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test, weights_train, weights_test = data_loader.split_data(X, y)

    # Further split training data for validation WITH WEIGHTS
    val_size = 0.2
    val_samples = int(len(X_train) * val_size)

    X_val = X_train[:val_samples]
    y_val = y_train[:val_samples]
    weights_val = weights_train[:val_samples]

    X_train_final = X_train[val_samples:]
    y_train_final = y_train[val_samples:]
    weights_train_final = weights_train[val_samples:]

    logger.info(f"Final split - Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    logger.info(f"Weight distribution after validation split:")
    logger.info(f"  Train final: {np.sum(weights_train_final == 2.0)} high-weight / {len(weights_train_final)} total")
    logger.info(f"  Validation: {np.sum(weights_val == 2.0)} high-weight / {len(weights_val)} total")
    logger.info(f"  Test: {np.sum(weights_test == 2.0)} high-weight / {len(weights_test)} total")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS WITH SAMPLE WEIGHTS")  # UPDATED MESSAGE
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
        weights_train_final, weights_val,
        feature_names, feature_types
    )

    # Check if any models were successfully trained
    successful_models = {k: v for k, v in all_results.items() if 'error' not in v}

    if not successful_models:
        logger.error("=" * 60)
        logger.error("CRITICAL ERROR: No models were successfully trained!")
        logger.error("=" * 60)
        logger.error("\nErrors encountered:")
        for model_name, result in all_results.items():
            if 'error' in result:
                logger.error(f"  {model_name}: {result['error']}")

        logger.error("\nPossible causes:")
        logger.error("1. Missing model configuration files in config/models/")
        logger.error("2. Missing or incompatible dependencies")
        logger.error("3. Data format issues")
        logger.error("\nPlease check the logs above for detailed error messages.")

        # Still save what we have for debugging
        partial_results = {
            'comparison': pd.DataFrame(),
            'model_results': all_results,
            'test_results': {},
            'best_model': None,
            'config': config
        }
        save_results(partial_results, config.get('output', {}).get('results_dir', 'results'))
        return partial_results

    logger.info(f"\nSuccessfully trained {len(successful_models)} out of {len(all_results)} models")
    if len(successful_models) < len(all_results):
        failed_models = [k for k in all_results.keys() if 'error' in all_results[k]]
        logger.warning(f"Failed models: {failed_models}")

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    test_results = {}
    for model_name, model_results in successful_models.items():
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
            test_metrics = evaluator.evaluate(y_test_np, test_predictions, weights_test)

            test_results[model_name] = {
                'test_metrics': test_metrics,
                'test_predictions': test_predictions
            }

            # Update results
            all_results[model_name]['test_metrics'] = test_metrics
            all_results[model_name]['test_predictions'] = test_predictions

            primary_metric = config.get('evaluation', {}).get('primary_metric', 'rmse')
            logger.info(f"{model_name} - Weighted Test {primary_metric.upper()}: {test_metrics.get(primary_metric, 0):.4f}")  # NEW LOGGING

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
            logger.error(traceback.format_exc())

    # Create model comparison
    logger.info("\n" + "=" * 60)
    logger.info("CREATING COMPARISON")
    logger.info("=" * 60)

    # Only use successful models for comparison
    comparison_df = evaluator.evaluate_models(successful_models)

    if comparison_df.empty:
        logger.warning("No models to compare - comparison DataFrame is empty")
    else:
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string(index=False))

        # Save comparison
        comparison_path = config.get('output', {}).get('comparison_file', 'results/model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")

    # ========================================
    # COMPREHENSIVE VISUALIZATIONS FOR ALL MODELS
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("CREATING VISUALIZATIONS FOR ALL MODELS")
    logger.info("=" * 60)

    try:
        # 1. Overall model comparison plot
        if not comparison_df.empty:
            logger.info("Creating overall model comparison plot...")
            primary_metric = config.get('evaluation', {}).get('primary_metric', 'rmse')

            # Plot comparison for primary metric
            visualizer.plot_model_comparison(comparison_df, primary_metric)

            # Also plot comparison for other important metrics
            secondary_metrics = config.get('evaluation', {}).get('secondary_metrics', ['mae', 'r2'])
            for metric in secondary_metrics:
                if f'val_{metric}' in comparison_df.columns or f'train_{metric}' in comparison_df.columns:
                    visualizer.plot_model_comparison(comparison_df, metric)

        # 2. Individual model visualizations
        logger.info("\nCreating individual model visualizations...")

        # Create a subdirectory for each model's plots
        models_plot_dir = Path(config.get('output', {}).get('plots_dir', 'results/plots')) / 'models'
        models_plot_dir.mkdir(parents=True, exist_ok=True)

        for model_name in successful_models.keys():
            logger.info(f"\n--- Visualizing {model_name} ---")

            # Create model-specific subdirectory
            model_plot_dir = models_plot_dir / model_name
            model_plot_dir.mkdir(parents=True, exist_ok=True)

            # Temporarily update visualizer output directory
            original_output_dir = visualizer.output_dir
            visualizer.output_dir = model_plot_dir

            # Create all visualizations for this model
            visualize_model_results(
                model_name,
                successful_models[model_name],
                test_results,
                y_test_np,
                visualizer,
                config
            )

            # Restore original output directory
            visualizer.output_dir = original_output_dir

        # 3. Comparative visualizations
        logger.info("\n--- Creating comparative visualizations ---")

        if len(successful_models) > 1 and not comparison_df.empty:
            # Create a comparison of training times
            from src.visualization_extensions import create_comparative_plots
            create_comparative_plots(
                comparison_df,
                test_results,
                y_test_np,
                config.get('output', {}).get('plots_dir', 'results/plots')
            )

        # 4. Best model detailed analysis
        best_model_name = None
        if successful_models:
            try:
                primary_metric = config.get('evaluation', {}).get('primary_metric', 'rmse')
                best_model_name = trainer.get_best_model(primary_metric, use_validation=True)

                # Create a special "best_model" directory with comprehensive analysis
                best_model_dir = Path(config.get('output', {}).get('plots_dir', 'results/plots')) / 'best_model'
                best_model_dir.mkdir(parents=True, exist_ok=True)

                original_output_dir = visualizer.output_dir
                visualizer.output_dir = best_model_dir

                if best_model_name in test_results:
                    logger.info(f"Creating detailed analysis for best model: {best_model_name}")

                    # All standard plots for best model
                    visualize_model_results(
                        best_model_name,
                        successful_models[best_model_name],
                        test_results,
                        y_test_np,
                        visualizer,
                        config
                    )

                    # Additional detailed analysis for best model
                    from src.visualization_extensions import create_detailed_best_model_analysis
                    create_detailed_best_model_analysis(
                        best_model_name,
                        test_results[best_model_name],
                        y_test_np,
                        best_model_dir
                    )

            except ValueError as e:
                logger.warning(f"Could not determine best model: {e}")

        # 5. Summary report with all plots referenced
        logger.info("\nCreating visualization summary report...")
        create_visualization_summary_report(
            successful_models,
            test_results,
            comparison_df,
            config.get('output', {}).get('plots_dir', 'results/plots')
        )

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.error(traceback.format_exc())

    # Save all results
    final_results = {
        'comparison': comparison_df if not comparison_df.empty else pd.DataFrame(),
        'model_results': all_results,
        'test_results': test_results,
        'best_model': trainer.get_best_model() if successful_models else None,
        'config': config,
        'successful_models': len(successful_models),
        'total_models_attempted': len(all_results),
        'used_sample_weights': True,  # NEW FIELD
        'weight_info': {  # NEW FIELD
            'total_samples': len(X),
            'high_weight_samples': int(np.sum(data_loader.sample_weights == 2.0)),
            'weight_scheme': 'quarter_based_double_weight_for_second_quarter'
        }
    }

    save_results(final_results, config.get('output', {}).get('results_dir', 'results'))

    logger.info("=" * 60)
    if successful_models:
        logger.info(f"PIPELINE COMPLETED - {len(successful_models)} models trained successfully with weighted evaluation")  # NEW LOGGING
        logger.info(f"Sample weights applied: Second quarter ({len(X)//4} samples) had 2x weight in error calculations")
    else:
        logger.info("PIPELINE COMPLETED WITH ERRORS - No models trained successfully")
    logger.info("=" * 60)

    return final_results


def create_visualization_summary_report(successful_models: dict,
                                       test_results: dict,
                                       comparison_df: pd.DataFrame,
                                       plots_dir: str) -> None:
    """
    Create a summary report of all visualizations created

    Args:
        successful_models: Dictionary of successful model results
        test_results: Dictionary of test results
        comparison_df: Model comparison DataFrame
        plots_dir: Directory containing plots
    """
    try:
        plots_path = Path(plots_dir)
        report_path = plots_path / 'visualization_summary.md'

        with open(report_path, 'w') as f:
            f.write("# Visualization Summary Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n")
            f.write(f"- Total models visualized: {len(successful_models)}\n")
            f.write(f"- Models with test results: {len(test_results)}\n\n")

            f.write("## Available Visualizations\n\n")

            f.write("### 1. Overall Comparisons\n")
            f.write("- `model_comparison_*.png`: Comparative bar charts for all metrics\n\n")

            f.write("### 2. Individual Model Plots\n")
            for model_name in successful_models.keys():
                f.write(f"\n#### {model_name}\n")
                model_dir = plots_path / 'models' / model_name
                if model_dir.exists():
                    plots = list(model_dir.glob('*.png'))
                    for plot in plots:
                        f.write(f"- `{plot.name}`\n")

            f.write("\n### 3. Best Model Analysis\n")
            best_model_dir = plots_path / 'best_model'
            if best_model_dir.exists():
                plots = list(best_model_dir.glob('*.png'))
                for plot in plots:
                    f.write(f"- `{plot.name}`\n")

            f.write("\n## Model Performance Summary\n\n")
            if not comparison_df.empty:
                f.write("```\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n```\n")

        logger.info(f"Visualization summary report saved to {report_path}")

    except Exception as e:
        logger.error(f"Error creating visualization summary: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to configuration file")

    args = parser.parse_args()

    try:
        results = main(args.config)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)