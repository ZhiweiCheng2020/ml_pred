
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
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), ncol=2)
        plt.title('Multi-Metric Model Comparison\n(All metrics normalized: higher is better)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.savefig(output_path / 'model_radar_chart.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating radar chart: {str(e)}")


def create_prediction_distribution_comparison(test_results: Dict[str, Dict],
                                              y_test: np.ndarray,
                                              output_path: Path) -> None:
    """Compare prediction distributions across models"""
    try:
        if not test_results:
            return

        n_models = len(test_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True)

        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (model_name, results) in enumerate(test_results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            predictions = results.get('test_predictions', [])

            if len(predictions) > 0:
                # Create 2D histogram
                h = ax.hist2d(y_test, predictions, bins=50, cmap='YlOrRd', alpha=0.8)
                plt.colorbar(h[3], ax=ax, label='Count')

                # Add diagonal line
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, alpha=0.7)

                # Calculate and display R²
                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

                ax.set_xlabel('Actual Values', fontsize=10)
                ax.set_ylabel('Predicted Values', fontsize=10)
                ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Prediction Distribution Comparison Across Models', fontsize=14, fontweight='bold')
        plt.savefig(output_path / 'prediction_distributions.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating prediction distribution comparison: {str(e)}")


def create_residual_comparison(test_results: Dict[str, Dict],
                               y_test: np.ndarray,
                               output_path: Path) -> None:
    """Compare residual patterns across models"""
    try:
        if not test_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

        # Collect residuals for all models
        all_residuals = {}
        for model_name, results in test_results.items():
            predictions = results.get('test_predictions', [])
            if len(predictions) > 0:
                all_residuals[model_name] = y_test - predictions

        if not all_residuals:
            return

        # Plot 1: Residual distributions (violin plot)
        ax1 = axes[0, 0]
        residual_df = pd.DataFrame(all_residuals)
        residual_df_melted = residual_df.melt(var_name='Model', value_name='Residual')

        sns.violinplot(data=residual_df_melted, x='Model', y='Residual', ax=ax1, inner='box', cut=0)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title('Residual Distribution Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Residual statistics comparison
        ax2 = axes[0, 1]
        stats_data = []
        for model_name, residuals in all_residuals.items():
            stats_data.append({
                'Model': model_name,
                'Mean': np.mean(residuals),
                'Std': np.std(residuals),
                'Skew': stats.skew(residuals),
                'Kurtosis': stats.kurtosis(residuals)
            })

        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('Model', inplace=True)

        # Create grouped bar plot
        stats_df[['Mean', 'Std']].plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title('Residual Statistics Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=10)
        ax2.legend(title='Statistic')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Q-Q plots comparison
        ax3 = axes[1, 0]
        for model_name, residuals in all_residuals.items():
            (osm, osr), (slope, intercept, r_value) = stats.probplot(residuals, dist="norm", plot=None)
            ax3.scatter(osm, osr, alpha=0.5, s=1, label=f'{model_name} (R²={r_value ** 2:.3f})')

        # Add reference line
        ax3.plot([-3, 3], [-3, 3], 'r--', lw=2, alpha=0.7, label='Normal')
        ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax3.set_ylabel('Sample Quantiles', fontsize=10)
        ax3.set_title('Q-Q Plot Comparison', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-3, 3)

        # Plot 4: Residual autocorrelation
        ax4 = axes[1, 1]

        for model_name, residuals in all_residuals.items():
            # Simple autocorrelation calculation
            lags = range(1, min(31, len(residuals) // 4))
            autocorr = [pd.Series(residuals).autocorr(lag) for lag in lags]
            ax4.plot(lags, autocorr, 'o-', alpha=0.7, label=model_name, markersize=4)

        ax4.axhline(y=0, color='r', linestyle='--', lw=1, alpha=0.5)
        ax4.set_xlabel('Lag', fontsize=10)
        ax4.set_ylabel('Autocorrelation', fontsize=10)
        ax4.set_title('Residual Autocorrelation Comparison', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Residual Analysis Across Models', fontsize=14, fontweight='bold')
        plt.savefig(output_path / 'residual_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating residual comparison: {str(e)}")


def create_model_ranking_heatmap(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing model rankings across different metrics"""
    try:
        # Select metrics for ranking
        metric_columns = [col for col in comparison_df.columns if
                          any(prefix in col for prefix in ['train_', 'val_', 'test_'])]

        if not metric_columns:
            return

        # Create ranking DataFrame
        ranking_df = pd.DataFrame(index=comparison_df['model'])

        for col in metric_columns:
            if col in comparison_df.columns:
                # Determine if lower or higher is better
                if any(metric in col for metric in ['rmse', 'mae', 'mse', 'mape']):
                    # Lower is better
                    ranking_df[col] = comparison_df[col].rank()
                elif any(metric in col for metric in ['r2']):
                    # Higher is better
                    ranking_df[col] = comparison_df[col].rank(ascending=False)
                else:
                    # Default: lower is better
                    ranking_df[col] = comparison_df[col].rank()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        # Create custom colormap (green=good=1, red=bad=n_models)
        sns.heatmap(ranking_df, annot=True, fmt='.0f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'Rank (1=Best)'}, ax=ax,
                    linewidths=1, linecolor='gray')

        ax.set_title('Model Ranking Heatmap Across All Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Models', fontsize=12)

        # Rotate x labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.savefig(output_path / 'model_ranking_heatmap.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating model ranking heatmap: {str(e)}")


def create_detailed_best_model_analysis(model_name: str,
                                        test_results: Dict,
                                        y_test: np.ndarray,
                                        output_dir: Path) -> None:
    """Create detailed analysis plots for the best performing model"""
    try:
        predictions = test_results.get('test_predictions', [])
        if len(predictions) == 0:
            return

        residuals = y_test - predictions

        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 16), constrained_layout=True)
        gs = fig.add_gridspec(4, 3)

        # 1. Prediction vs Actual with confidence bands
        ax1 = fig.add_subplot(gs[0, :2])
        sorted_indices = np.argsort(y_test)
        sorted_actual = y_test[sorted_indices]
        sorted_pred = predictions[sorted_indices]

        ax1.scatter(range(len(sorted_actual)), sorted_actual, alpha=0.5, s=1, label='Actual', color='blue')
        ax1.scatter(range(len(sorted_pred)), sorted_pred, alpha=0.5, s=1, label='Predicted', color='red')

        # Add rolling average
        window = max(10, len(sorted_actual) // 50)
        rolling_actual = pd.Series(sorted_actual).rolling(window, center=True).mean()
        rolling_pred = pd.Series(sorted_pred).rolling(window, center=True).mean()
        ax1.plot(range(len(rolling_actual)), rolling_actual, 'b-', linewidth=2, alpha=0.7)
        ax1.plot(range(len(rolling_pred)), rolling_pred, 'r-', linewidth=2, alpha=0.7)

        ax1.set_xlabel('Sample Index (sorted by actual value)', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title(f'{model_name}: Predictions vs Actuals (Sorted)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Error distribution by value range
        ax2 = fig.add_subplot(gs[0, 2])
        n_bins = min(10, len(y_test) // 100)
        try:
            value_bins = pd.qcut(y_test, q=n_bins, duplicates='drop')
            error_by_bin = pd.DataFrame({'bin': value_bins, 'error': np.abs(residuals)})
            error_by_bin.boxplot(column='error', by='bin', ax=ax2)
            ax2.set_xlabel('Actual Value Range', fontsize=11)
            ax2.set_ylabel('Absolute Error', fontsize=11)
            ax2.set_title('Error Distribution by Value Range', fontsize=12, fontweight='bold')
            ax2.set_xticklabels([])
            plt.sca(ax2)
            plt.xticks(rotation=45)
        except:
            ax2.text(0.5, 0.5, 'Not enough data for binning', ha='center', va='center')

        # 3. Residual histogram with fitted distribution
        ax3 = fig.add_subplot(gs[1, 0])
        n, bins, patches = ax3.hist(residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

        # Fit normal distribution
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal fit: μ={mu:.4f}, σ={std:.4f}')

        ax3.set_xlabel('Residuals', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Residual Distribution with Normal Fit', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative error distribution
        ax4 = fig.add_subplot(gs[1, 1])
        sorted_errors = np.sort(np.abs(residuals))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        ax4.plot(sorted_errors, cumulative, 'b-', linewidth=2)
        ax4.fill_between(sorted_errors, 0, cumulative, alpha=0.3)

        # Add percentile markers
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(np.abs(residuals), p)
            ax4.axvline(x=val, color='r', linestyle='--', alpha=0.5)
            ax4.text(val, 0.05, f'{p}%', rotation=90, fontsize=9)

        ax4.set_xlabel('Absolute Error', fontsize=11)
        ax4.set_ylabel('Cumulative Probability', fontsize=11)
        ax4.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. Scatter plot with hexbin
        ax5 = fig.add_subplot(gs[1, 2])

        # Sample for visibility if too many points
        if len(predictions) > 5000:
            sample_idx = np.random.choice(len(predictions), 5000, replace=False)
            sample_pred = predictions[sample_idx]
            sample_actual = y_test[sample_idx]
        else:
            sample_pred = predictions
            sample_actual = y_test

        ax5.hexbin(sample_actual, sample_pred, gridsize=30, cmap='YlOrRd', alpha=0.8)

        # Add perfect prediction line
        min_val = min(sample_actual.min(), sample_pred.min())
        max_val = max(sample_actual.max(), sample_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, alpha=0.7)

        ax5.set_xlabel('Actual Values', fontsize=11)
        ax5.set_ylabel('Predicted Values', fontsize=11)
        ax5.set_title('Prediction Accuracy Heatmap', fontsize=12, fontweight='bold')

        # 6. Time series of errors
        ax6 = fig.add_subplot(gs[2, :])
        ax6.plot(np.abs(residuals), alpha=0.5, linewidth=0.5, color='gray')

        # Add moving average
        window = max(10, len(residuals) // 50)
        moving_avg = pd.Series(np.abs(residuals)).rolling(window, center=True).mean()
        ax6.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')

        ax6.set_xlabel('Sample Index', fontsize=11)
        ax6.set_ylabel('Absolute Error', fontsize=11)
        ax6.set_title('Error Evolution Across Test Set', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Performance metrics summary
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')

        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Handle MAPE calculation
        if np.any(y_test == 0):
            non_zero_mask = y_test != 0
            if np.any(non_zero_mask):
                mape = np.mean(
                    np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = np.nan
        else:
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        max_error = np.max(np.abs(residuals))

        # Create metrics table
        metrics_text = f"""
        Performance Metrics Summary for {model_name}:

        RMSE: {rmse:.6f}          MAE: {mae:.6f}          R²: {r2:.6f}
        MAPE: {mape:.2f}%         Max Error: {max_error:.6f}

        Residual Statistics:
        Mean: {np.mean(residuals):.6f}     Std: {np.std(residuals):.6f}
        Skewness: {stats.skew(residuals):.6f}     Kurtosis: {stats.kurtosis(residuals):.6f}

        Error Percentiles:
        50th: {np.percentile(np.abs(residuals), 50):.6f}
        75th: {np.percentile(np.abs(residuals), 75):.6f}
        90th: {np.percentile(np.abs(residuals), 90):.6f}
        95th: {np.percentile(np.abs(residuals), 95):.6f}
        99th: {np.percentile(np.abs(residuals), 99):.6f}
        """

        ax7.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=ax7.transAxes)

        plt.suptitle(f'Comprehensive Analysis: {model_name} (Best Model)',
                     fontsize=16, fontweight='bold')

        plt.savefig(output_dir / f'{model_name}_detailed_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating detailed best model analysis: {str(e)}")


def create_error_analysis_by_feature_range(model_name: str,
                                           predictions: np.ndarray,
                                           y_test: np.ndarray,
                                           X_test: np.ndarray,
                                           feature_names: list,
                                           output_dir: Path,
                                           top_n_features: int = 4) -> None:
    """
    Analyze prediction errors based on feature value ranges

    Args:
        model_name: Name of the model
        predictions: Model predictions
        y_test: True test values
        X_test: Test features
        feature_names: List of feature names
        output_dir: Output directory
        top_n_features: Number of top features to analyze
    """
    try:
        residuals = y_test - predictions
        abs_residuals = np.abs(residuals)

        # Calculate correlation between features and absolute errors
        error_correlations = []
        for i, feature_name in enumerate(feature_names[:min(len(feature_names), 100)]):  # Limit to first 100 features
            corr = np.corrcoef(X_test[:, i], abs_residuals)[0, 1]
            error_correlations.append((feature_name, abs(corr)))

        # Sort by correlation and take top N
        error_correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = error_correlations[:top_n_features]

        # Create subplot for each top feature
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        axes = axes.flatten()

        for idx, (feature_name, corr_value) in enumerate(top_features):
            if idx >= len(axes):
                break

            ax = axes[idx]
            feature_idx = feature_names.index(feature_name)
            feature_values = X_test[:, feature_idx]

            # Create bins for feature values
            try:
                bins = pd.qcut(feature_values, q=5, duplicates='drop')
                bin_labels = [f'Q{i + 1}' for i in range(len(bins.unique()))]

                # Calculate error statistics for each bin
                error_by_bin = pd.DataFrame({
                    'bin': bins,
                    'error': abs_residuals,
                    'feature_value': feature_values
                })

                # Create box plot
                error_by_bin.boxplot(column='error', by='bin', ax=ax)
                ax.set_xlabel(f'{feature_name} (Quintiles)', fontsize=10)
                ax.set_ylabel('Absolute Error', fontsize=10)
                ax.set_title(f'Error vs {feature_name}\nCorrelation: {corr_value:.3f}',
                             fontsize=11, fontweight='bold')
                plt.sca(ax)
                plt.xticks(rotation=45)

            except Exception as e:
                # Fallback to scatter plot if binning fails
                scatter = ax.scatter(feature_values, abs_residuals, alpha=0.5, s=5)
                ax.set_xlabel(feature_name, fontsize=10)
                ax.set_ylabel('Absolute Error', fontsize=10)
                ax.set_title(f'Error vs {feature_name}\nCorrelation: {corr_value:.3f}',
                             fontsize=11, fontweight='bold')

            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name}: Error Analysis by Top Correlated Features',
                     fontsize=14, fontweight='bold')

        plt.savefig(output_dir / f'{model_name}_error_by_features.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating feature-based error analysis: {str(e)}")


def create_prediction_confidence_bands(model_name: str,
                                       predictions: np.ndarray,
                                       y_test: np.ndarray,
                                       output_dir: Path) -> None:
    """
    Create prediction plots with confidence bands

    Args:
        model_name: Name of the model
        predictions: Model predictions
        y_test: True test values
        output_dir: Output directory
    """
    try:
        residuals = y_test - predictions

        # Calculate prediction intervals (assuming normal distribution of errors)
        residual_std = np.std(residuals)

        # Sort by predictions for better visualization
        sorted_indices = np.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]
        sorted_actual = y_test[sorted_indices]

        # Calculate confidence bands (95% confidence interval)
        confidence_lower = sorted_predictions - 1.96 * residual_std
        confidence_upper = sorted_predictions + 1.96 * residual_std

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        # Plot actual vs predicted
        ax.scatter(range(len(sorted_actual)), sorted_actual, alpha=0.6, s=5,
                   label='Actual', color='blue')
        ax.plot(range(len(sorted_predictions)), sorted_predictions, 'r-',
                linewidth=2, label='Predicted', alpha=0.8)

        # Add confidence bands
        ax.fill_between(range(len(sorted_predictions)),
                        confidence_lower, confidence_upper,
                        alpha=0.3, color='red', label='95% Confidence Band')

        # Calculate percentage within bands
        within_bands = np.sum((sorted_actual >= confidence_lower) &
                              (sorted_actual <= confidence_upper))
        percentage_within = (within_bands / len(sorted_actual)) * 100

        ax.set_xlabel('Sample Index (sorted by prediction)', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{model_name}: Predictions with Confidence Bands\n' +
                     f'{percentage_within:.1f}% of actual values within 95% confidence band',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(output_dir / f'{model_name}_confidence_bands.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating confidence bands plot: {str(e)}")


def create_model_comparison_matrix(comparison_df: pd.DataFrame,
                                   output_path: Path) -> None:
    """
    Create a matrix of scatter plots comparing all models pairwise

    Args:
        comparison_df: Model comparison DataFrame
        output_path: Output directory
    """
    try:
        metrics = ['val_rmse', 'val_r2', 'val_mae', 'training_time']
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        if len(available_metrics) < 2:
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(15, 15), constrained_layout=True)

        for i in range(n_metrics):
            for j in range(n_metrics):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(comparison_df[available_metrics[i]], bins=10, alpha=0.7, color='blue')
                    ax.set_xlabel(available_metrics[i], fontsize=9)
                    if i == 0:
                        ax.set_ylabel('Count', fontsize=9)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(comparison_df[available_metrics[j]],
                               comparison_df[available_metrics[i]],
                               s=100, alpha=0.7)

                    # Add model labels
                    for idx, row in comparison_df.iterrows():
                        ax.annotate(row['model'][:3],  # Use first 3 letters for space
                                    (row[available_metrics[j]], row[available_metrics[i]]),
                                    fontsize=7, alpha=0.8)

                    if i == n_metrics - 1:
                        ax.set_xlabel(available_metrics[j], fontsize=9)
                    if j == 0:
                        ax.set_ylabel(available_metrics[i], fontsize=9)

                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

        plt.suptitle('Model Comparison Matrix', fontsize=14, fontweight='bold')
        plt.savefig(output_path / 'model_comparison_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating model comparison matrix: {str(e)}")


def create_performance_summary_dashboard(comparison_df: pd.DataFrame,
                                         test_results: Dict[str, Dict],
                                         y_test: np.ndarray,
                                         output_path: Path) -> None:
    """
    Create a comprehensive dashboard summarizing all models' performance

    Args:
        comparison_df: Model comparison DataFrame
        test_results: Test results for all models
        y_test: True test values
        output_path: Output directory
    """
    try:
        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 4)

        # 1. Overall ranking (top-left, 2x2)
        ax1 = fig.add_subplot(gs[:2, :2])
        if 'val_rmse' in comparison_df.columns:
            sorted_df = comparison_df.sort_values('val_rmse')
            y_pos = np.arange(len(sorted_df))
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_df)))

            bars = ax1.barh(y_pos, sorted_df['val_rmse'], color=colors)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(sorted_df['model'])
            ax1.set_xlabel('Validation RMSE', fontsize=11)
            ax1.set_title('Model Ranking by RMSE', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, sorted_df['val_rmse'])):
                ax1.text(value, bar.get_y() + bar.get_height() / 2,
                         f'{value:.4f}', ha='left', va='center', fontsize=9)

        # 2. Training time comparison (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'training_time' in comparison_df.columns:
            sorted_time = comparison_df.sort_values('training_time')
            ax2.bar(range(len(sorted_time)), sorted_time['training_time'],
                    color='skyblue', edgecolor='navy')
            ax2.set_xticks(range(len(sorted_time)))
            ax2.set_xticklabels(sorted_time['model'], rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Time (seconds)', fontsize=10)
            ax2.set_title('Training Time', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Feature usage (top-right)
        ax3 = fig.add_subplot(gs[0, 3])
        if 'n_features' in comparison_df.columns:
            sorted_features = comparison_df.sort_values('n_features')
            ax3.bar(range(len(sorted_features)), sorted_features['n_features'],
                    color='lightgreen', edgecolor='darkgreen')
            ax3.set_xticks(range(len(sorted_features)))
            ax3.set_xticklabels(sorted_features['model'], rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('# Features', fontsize=10)
            ax3.set_title('Feature Count', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. R² comparison (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'val_r2' in comparison_df.columns:
            sorted_r2 = comparison_df.sort_values('val_r2', ascending=False)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_r2)))
            ax4.bar(range(len(sorted_r2)), sorted_r2['val_r2'], color=colors)
            ax4.set_xticks(range(len(sorted_r2)))
            ax4.set_xticklabels(sorted_r2['model'], rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel('R² Score', fontsize=10)
            ax4.set_title('R² Score Comparison', fontsize=11, fontweight='bold')
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='R²=0.5')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()

        # 5. Summary statistics table (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Create summary table
        summary_data = []
        for model in comparison_df['model']:
            row_data = [model]
            row = comparison_df[comparison_df['model'] == model].iloc[0]

            for metric in ['val_rmse', 'val_mae', 'val_r2', 'training_time', 'n_features']:
                if metric in comparison_df.columns:
                    value = row[metric]
                    if metric == 'training_time':
                        row_data.append(f'{value:.1f}s')
                    elif metric == 'n_features':
                        row_data.append(f'{int(value)}')
                    else:
                        row_data.append(f'{value:.4f}')
                else:
                    row_data.append('N/A')

            summary_data.append(row_data)

        # Create table
        col_labels = ['Model', 'Val RMSE', 'Val MAE', 'Val R²', 'Time', 'Features']
        table = ax5.table(cellText=summary_data, colLabels=col_labels,
                          cellLoc='center', loc='center',
                          colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style the table
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code the cells based on performance
        if 'val_rmse' in comparison_df.columns:
            rmse_values = comparison_df['val_rmse'].values
            rmse_norm = (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min() + 1e-10)

            for i, model in enumerate(comparison_df['model']):
                color_intensity = rmse_norm[i]
                color = plt.cm.RdYlGn_r(color_intensity * 0.5 + 0.25)
                table[(i + 1, 1)].set_facecolor(color)

        plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(output_path / 'performance_dashboard.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error creating performance dashboard: {str(e)}")