
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="darkgrid", palette="husl")


def create_comparative_plots(comparison_df: pd.DataFrame,
                             test_results: Dict[str, Dict],
                             y_test: np.ndarray,
                             output_dir: str) -> None:
    """
    Create comparative visualizations across all models

    Args:
        comparison_df: Model comparison DataFrame
        test_results: Test results for all models
        y_test: True test values
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir) / 'comparative'
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Training Time vs Performance Scatter
    create_time_vs_performance_plot(comparison_df, output_path)

    # 2. Feature Count vs Performance
    create_features_vs_performance_plot(comparison_df, output_path)

    # 3. Multi-metric Radar Chart
    create_radar_chart(comparison_df, output_path)

    # 4. Prediction Distribution Comparison
    create_prediction_distribution_comparison(test_results, y_test, output_path)

    # 5. Residual Comparison Across Models
    create_residual_comparison(test_results, y_test, output_path)

    # 6. Model Ranking Heatmap
    create_model_ranking_heatmap(comparison_df, output_path)


def create_time_vs_performance_plot(comparison_df: pd.DataFrame,
                                    output_path: Path) -> None:
    """Create scatter plot of training time vs performance"""
    try:
        if 'training_time' not in comparison_df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

        # Plot 1: Time vs RMSE
        ax1 = axes[0]
        if 'val_rmse' in comparison_df.columns:
            scatter = ax1.scatter(comparison_df['training_time'],
                                  comparison_df['val_rmse'],
                                  s=200, alpha=0.6, c=range(len(comparison_df)),
                                  cmap='viridis', edgecolors='black', linewidth=2)

            # Add model labels
            for idx, row in comparison_df.iterrows():
                ax1.annotate(row['model'],
                             (row['training_time'], row['val_rmse']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')

            ax1.set_xlabel('Training Time (seconds)', fontsize=12)
            ax1.set_ylabel('Validation RMSE', fontsize=12)
            ax1.set_title('Training Efficiency: Time vs Performance', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Add efficiency frontier
            sorted_df = comparison_df.sort_values('training_time')
            min_rmse = sorted_df['val_rmse'].cummin()
            ax1.plot(sorted_df['training_time'], min_rmse, 'r--', alpha=0.5,
                     linewidth=2, label='Efficiency Frontier')
            ax1.legend()

        # Plot 2: Time vs R²
        ax2 = axes[1]
        if 'val_r2' in comparison_df.columns:
            scatter = ax2.scatter(comparison_df['training_time'],
                                  comparison_df['val_r2'],
                                  s=200, alpha=0.6, c=range(len(comparison_df)),
                                  cmap='plasma', edgecolors='black', linewidth=2)

            for idx, row in comparison_df.iterrows():
                ax2.annotate(row['model'],
                             (row['training_time'], row['val_r2']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')

            ax2.set_xlabel('Training Time (seconds)', fontsize=12)
            ax2.set_ylabel('Validation R²', fontsize=12)
            ax2.set_title('Training Efficiency: Time vs R² Score', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Model Training Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(output_path / 'time_vs_performance.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating time vs performance plot: {str(e)}")


def create_features_vs_performance_plot(comparison_df: pd.DataFrame,
                                        output_path: Path) -> None:
    """Create plot of feature count vs model performance"""
    try:
        if 'n_features' not in comparison_df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

        if 'val_rmse' in comparison_df.columns:
            # Create bubble plot
            sizes = 500 * (1 / comparison_df['val_rmse'])  # Inverse RMSE for size
            colors = comparison_df['training_time'] if 'training_time' in comparison_df.columns else range(
                len(comparison_df))

            scatter = ax.scatter(comparison_df['n_features'],
                                 comparison_df['val_rmse'],
                                 s=sizes, alpha=0.6, c=colors,
                                 cmap='coolwarm', edgecolors='black', linewidth=2)

            # Add model labels
            for idx, row in comparison_df.iterrows():
                ax.annotate(row['model'],
                            (row['n_features'], row['val_rmse']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')

            ax.set_xlabel('Number of Features Used', fontsize=12)
            ax.set_ylabel('Validation RMSE', fontsize=12)
            ax.set_title('Feature Dimensionality vs Model Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Time (s)' if 'training_time' in comparison_df.columns else 'Model Index',
                           fontsize=10)

            # Add size legend
            handles, labels = [], []
            for rmse in [comparison_df['val_rmse'].min(), comparison_df['val_rmse'].median(),
                         comparison_df['val_rmse'].max()]:
                handles.append(plt.scatter([], [], s=500 / rmse, c='gray', alpha=0.6))
                labels.append(f'RMSE: {rmse:.4f}')
            ax.legend(handles, labels, scatterpoints=1, frameon=True,
                      labelspacing=2, title='Bubble Size', loc='upper right')

        plt.savefig(output_path / 'features_vs_performance.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating features vs performance plot: {str(e)}")


def create_radar_chart(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Create radar chart comparing multiple metrics across models"""
    try:
        # Select metrics for radar chart
        metrics = []
        metric_names = []

        if 'val_rmse' in comparison_df.columns:
            metrics.append('val_rmse')
            metric_names.append('RMSE (inverted)')
        if 'val_mae' in comparison_df.columns:
            metrics.append('val_mae')
            metric_names.append('MAE (inverted)')
        if 'val_r2' in comparison_df.columns:
            metrics.append('val_r2')
            metric_names.append('R²')
        if 'training_time' in comparison_df.columns:
            metrics.append('training_time')
            metric_names.append('Speed (inverted time)')
        if 'n_features' in comparison_df.columns:
            metrics.append('n_features')
            metric_names.append('Feature Efficiency')

        if len(metrics) < 3:
            return

        # Normalize metrics (higher is better)
        normalized_df = comparison_df.copy()
        for metric in metrics:
            if metric in ['val_rmse', 'val_mae', 'training_time']:
                # Invert: lower is better -> higher is better
                normalized_df[metric] = 1 / (1 + normalized_df[metric])
            elif metric == 'n_features':
                # Inverse normalize: fewer features is better
                normalized_df[metric] = 1 / (1 + normalized_df[metric] / normalized_df[metric].max())
            elif metric == 'val_r2':
                # Already higher is better, just ensure [0, 1]
                normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / \
                                        (normalized_df[metric].max() - normalized_df[metric].min() + 1e-10)

        # Create radar chart
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='polar')

        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(normalized_df)))

        for idx, (_, row) in enumerate(normalized_df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'],
                    color=colors[idx], markersize=8)
            ax.