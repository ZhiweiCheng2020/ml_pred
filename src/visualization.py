"""
Visualization utilities - Compatible with matplotlib 3.7+ and seaborn 0.13+
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import warnings

logger = logging.getLogger(__name__)

# Handle deprecation warnings for seaborn 0.13+
warnings.filterwarnings('ignore', category=FutureWarning)

# Set style for seaborn 0.13+
# Use the new style names (v0_8 prefix is deprecated)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    # Fallback for newer versions where the style name changed
    plt.style.use('seaborn-darkgrid') if 'seaborn-darkgrid' in plt.style.available else plt.style.use('default')

# Set palette using seaborn 0.13+ syntax
sns.set_theme(style="darkgrid", palette="husl")


class Visualizer:
    """Handle visualization of results with matplotlib 3.7+ and seaborn 0.13+"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Visualizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output', {}).get('plots_dir', 'results/plots'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default figure parameters for matplotlib 3.7+
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['savefig.bbox'] = 'tight'

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

        # Plot bars with matplotlib 3.7+ syntax
        train_bars = ax.bar(x - width / 2, comparison_df[train_col], width, label='Train')

        if val_col in comparison_df.columns:
            val_bars = ax.bar(x + width / 2, comparison_df[val_col], width, label='Validation')

        # Customize plot
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [train_bars, val_bars if val_col in comparison_df.columns else []]:
            if bars:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'model_comparison_{metric}.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Model comparison plot saved to {filepath}")

        plt.show()
        plt.close()

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, dataset_type: str = 'test',
                                   save: bool = True) -> None:
        """
        Plot predictions vs actual values with matplotlib 3.7+ features

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            dataset_type: Type of dataset
            save: Whether to save the plot
        """
        # Create figure with constrained layout (matplotlib 3.7+)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Scatter plot with enhanced visuals
        ax1 = axes[0]

        # Use hexbin for large datasets (better performance)
        if len(y_true) > 5000:
            hexbin = ax1.hexbin(y_true, y_pred, gridsize=50, cmap='YlOrRd', alpha=0.8)
            fig.colorbar(hexbin, ax=ax1, label='Count')
        else:
            scatter = ax1.scatter(y_true, y_pred, alpha=0.5, s=10, c='blue', edgecolors='none')

        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect Prediction')

        # Calculate R² for display
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)

        ax1.set_xlabel('Actual', fontsize=12)
        ax1.set_ylabel('Predicted', fontsize=12)
        ax1.set_title(f'{model_name} - Predictions vs Actual ({dataset_type})\nR² = {r2:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()

        # Residual plot with seaborn 0.13+ styling
        ax2 = axes[1]
        residuals = y_true - y_pred

        # Use violin plot for residual distribution (seaborn 0.13+)
        if len(residuals) > 100:
            # Create bins for violin plot
            pred_bins = pd.qcut(y_pred, q=min(10, len(y_pred)//100), duplicates='drop')
            df_resid = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals, 'Bins': pred_bins})

            sns.violinplot(data=df_resid, x='Bins', y='Residuals', ax=ax2, inner='box', cut=0)
            ax2.set_xticklabels([])
            ax2.set_xlabel('Predicted (Binned)', fontsize=12)
        else:
            ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
            ax2.set_xlabel('Predicted', fontsize=12)

        ax2.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title(f'{model_name} - Residual Plot ({dataset_type})', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle(f'Model Performance Analysis: {model_name}', fontsize=14, fontweight='bold', y=1.02)

        if save:
            filepath = self.output_dir / f'{model_name}_{dataset_type}_predictions.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Predictions plot saved to {filepath}")

        plt.show()
        plt.close()

    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, save: bool = True) -> None:
        """
        Plot residual distribution with matplotlib 3.7+ and seaborn 0.13+

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        residuals = y_true - y_pred

        # Create figure with matplotlib 3.7+ constrained layout
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Histogram with KDE overlay (seaborn 0.13+)
        ax1 = axes[0]

        # Use seaborn histplot (replaces distplot in 0.13+)
        sns.histplot(residuals, bins=50, kde=True, stat='density',
                    color='blue', alpha=0.7, ax=ax1, edgecolor='black', linewidth=0.5)

        ax1.axvline(x=0, color='r', linestyle='--', lw=2, alpha=0.7, label='Zero')
        ax1.axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, alpha=0.7,
                   label=f'Mean: {np.mean(residuals):.4f}')

        ax1.set_xlabel('Residuals', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'{model_name} - Residual Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()

        # Q-Q plot with scipy
        ax2 = axes[1]
        from scipy import stats

        # Enhanced Q-Q plot
        (osm, osr), (slope, intercept, r_value) = stats.probplot(residuals, dist="norm", plot=None)
        ax2.scatter(osm, osr, alpha=0.5, s=10, color='blue', edgecolors='none')

        # Add reference line
        ax2.plot(osm, slope * osm + intercept, 'r-', lw=2, alpha=0.7, label=f'R² = {r_value**2:.4f}')

        ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax2.set_ylabel('Sample Quantiles', fontsize=12)
        ax2.set_title(f'{model_name} - Q-Q Plot', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()

        plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold', y=1.02)

        if save:
            filepath = self.output_dir / f'{model_name}_residual_distribution.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Residual distribution plot saved to {filepath}")

        plt.show()
        plt.close()

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                                model_name: str, top_n: int = 20,
                                save: bool = True) -> None:
        """
        Plot feature importance with enhanced visuals

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

        # Create figure with matplotlib 3.7+ features
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

        # Create horizontal bar plot with gradient colors
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(top_features)))

        bars = ax.barh(y_pos, top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            ax.text(value, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        # Add colorbar to show importance scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=top_features['importance'].min(),
                                                     vmax=top_features['importance'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label('Importance Scale', fontsize=10)

        if save:
            filepath = self.output_dir / f'{model_name}_feature_importance.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Feature importance plot saved to {filepath}")

        plt.show()
        plt.close()

    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                             model_name: str, metric: str = 'rmse',
                             save: bool = True) -> None:
        """
        Plot learning curves with matplotlib 3.7+ features

        Args:
            train_scores: Training scores over epochs/iterations
            val_scores: Validation scores over epochs/iterations
            model_name: Name of the model
            metric: Metric name
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        epochs = range(1, len(train_scores) + 1)

        # Plot with enhanced styling
        ax.plot(epochs, train_scores, 'b-', label='Training', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_scores, 'r-', label='Validation', linewidth=2, alpha=0.8)

        # Add shaded area between curves
        ax.fill_between(epochs, train_scores, val_scores, alpha=0.2, color='gray')

        # Find best epoch
        best_epoch = np.argmin(val_scores) + 1
        best_val_score = min(val_scores)
        ax.plot(best_epoch, best_val_score, 'go', markersize=10,
               label=f'Best: Epoch {best_epoch} ({best_val_score:.4f})')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{model_name} - Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if save:
            filepath = self.output_dir / f'{model_name}_learning_curves.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Learning curves plot saved to {filepath}")

        plt.show()
        plt.close()