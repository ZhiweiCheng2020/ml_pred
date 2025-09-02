"""
Visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Handle visualization of results"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Visualizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output', {}).get('plots_dir', 'results/plots'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                              metric: str = 'rmse',
                              save: bool = True) -> None:
        """
        Plot model comparison bar chart

        Args:
            comparison_df: DataFrame with model comparison
            metric: Metric to plot
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data
        train_col = f'train_{metric}'
        val_col = f'val_{metric}'

        if train_col not in comparison_df.columns:
            logger.warning(f"Column {train_col} not found in comparison DataFrame")
            return

        x = np.arange(len(comparison_df))
        width = 0.35

        # Plot bars
        train_bars = ax.bar(x - width / 2, comparison_df[train_col], width, label='Train')

        if val_col in comparison_df.columns:
            val_bars = ax.bar(x + width / 2, comparison_df[val_col], width, label='Validation')

        # Customize plot
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'model_comparison_{metric}.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {filepath}")

        plt.show()

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, dataset_type: str = 'test',
                                   save: bool = True) -> None:
        """
        Plot predictions vs actual values

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            dataset_type: Type of dataset
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=10)

        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{model_name} - Predictions vs Actual ({dataset_type})')
        ax1.grid(True, alpha=0.3)

        # Residual plot
        ax2 = axes[1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)

        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name} - Residual Plot ({dataset_type})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{model_name}_{dataset_type}_predictions.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {filepath}")

        plt.show()

    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, save: bool = True) -> None:
        """
        Plot residual distribution

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{model_name} - Residual Distribution')
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'{model_name} - Q-Q Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{model_name}_residual_distribution.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Residual distribution plot saved to {filepath}")

        plt.show()

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                                model_name: str, top_n: int = 20,
                                save: bool = True) -> None:
        """
        Plot feature importance

        Args:
            feature_importance: DataFrame with feature importance
            model_name: Name of the model
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        if feature_importance is None or feature_importance.empty:
            logger.warning(f"No feature importance available for {model_name}")
            return

        # Get top features
        top_features = feature_importance.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{model_name}_feature_importance.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")

        plt.show()

    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                             model_name: str, metric: str = 'rmse',
                             save: bool = True) -> None:
        """
        Plot learning curves

        Args:
            train_scores: Training scores over epochs/iterations
            val_scores: Validation scores over epochs/iterations
            model_name: Name of the model
            metric: Metric name
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(train_scores) + 1)

        ax.plot(epochs, train_scores, 'b-', label='Training')
        ax.plot(epochs, val_scores, 'r-', label='Validation')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{model_name} - Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{model_name}_learning_curves.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {filepath}")

        plt.show()

    def create_summary_report(self, results: Dict[str, Any]) -> None:
        """
        Create a summary report with multiple visualizations

        Args:
            results: Dictionary with all results
        """
        logger.info("Creating summary report...")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Model comparison
        if 'comparison' in results:
            ax1 = plt.subplot(2, 2, 1)
            self._plot_model_comparison_subplot(results['comparison'], ax1)

        # Best model predictions
        if 'best_model' in results:
            best_model = results['best_model']
            if 'predictions' in best_model:
                ax2 = plt.subplot(2, 2, 2)
                self._plot_predictions_subplot(
                    best_model['y_true'],
                    best_model['predictions'],
                    best_model['name'],
                    ax2
                )

        # Feature importance
        if 'feature_importance' in results:
            ax3 = plt.subplot(2, 2, 3)
            self._plot_feature_importance_subplot(
                results['feature_importance'],
                ax3
            )

        # Metrics summary
        ax4 = plt.subplot(2, 2, 4)
        self._plot_metrics_summary(results.get('comparison'), ax4)

        plt.suptitle('ML Pipeline Summary Report', fontsize=16, y=1.02)
        plt.tight_layout()

        # Save report
        filepath = self.output_dir / 'summary_report.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        logger.info(f"Summary report saved to {filepath}")

        plt.show()

    def _plot_model_comparison_subplot(self, comparison_df: pd.DataFrame, ax) -> None:
        """Helper function for model comparison subplot"""
        if 'val_rmse' in comparison_df.columns:
            comparison_df.plot(x='model', y=['train_rmse', 'val_rmse'],
                               kind='bar', ax=ax)
        else:
            comparison_df.plot(x='model', y='train_rmse', kind='bar', ax=ax)

        ax.set_title('Model Comparison - RMSE')
        ax.set_xlabel('')
        ax.set_ylabel('RMSE')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_predictions_subplot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str, ax) -> None:
        """Helper function for predictions subplot"""
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name} - Predictions vs Actual')
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance_subplot(self, feature_importance: pd.DataFrame, ax) -> None:
        """Helper function for feature importance subplot"""
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Features')
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_metrics_summary(self, comparison_df: pd.DataFrame, ax) -> None:
        """Helper function for metrics summary"""
        if comparison_df is None:
            return

        # Create metrics summary table
        metrics_cols = [col for col in comparison_df.columns
                        if col.startswith('val_') or col.startswith('train_')]

        if not metrics_cols:
            return

        summary_data = comparison_df[['model'] + metrics_cols].round(4)

        # Create table
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=summary_data.values,
                         colLabels=summary_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title('Metrics Summary', pad=20)