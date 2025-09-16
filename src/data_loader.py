"""
Data loading and splitting utilities
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, Union
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and train-test splitting"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.random_state = self.data_config.get('random_state', 42)

        # Multi-output support
        self.multi_output_config = config.get('evaluation', {}).get('multi_output', {})
        self.is_multi_output = self.multi_output_config.get('enabled', False)
        self.n_outputs = self.multi_output_config.get('n_outputs', 1)

        self.X = None
        self.y = None
        self.feature_types = None
        self.feature_names = None
        self.sample_weights = None

    def create_sample_weights(self, n_samples: int) -> np.ndarray:
        """
        Create sample weights with double weight for second quarter of data

        Args:
            n_samples: Total number of samples

        Returns:
            Array of sample weights
        """
        weights = np.ones(n_samples, dtype=np.float64)

        # Calculate quarter boundaries
        quarter_size = n_samples // 4

        # Second quarter gets double weight (2.0)
        start_idx = quarter_size
        end_idx = 2 * quarter_size
        weights[start_idx:end_idx] = 2.0

        logger.info(f"Created sample weights: {n_samples} total samples")
        logger.info(f"- First quarter (0-{quarter_size}): weight=1.0")
        logger.info(f"- Second quarter ({start_idx}-{end_idx}): weight=2.0")
        logger.info(f"- Third quarter ({end_idx}-{3*quarter_size}): weight=1.0")
        logger.info(f"- Fourth quarter ({3*quarter_size}-{n_samples}): weight=1.0")
        logger.info(f"Total weighted samples: {np.sum(weights):.0f}")

        return weights

    def load_data(self) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series], Dict[str, str]]:
        """
        Load features, targets, feature types, and create sample weights

        Returns:
            Tuple of (features DataFrame, targets DataFrame/Series, feature_types dict)
        """
        logger.info("Loading data...")

        # Load features
        features_path = Path(self.data_config['features_path'])
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        self.X = pd.read_csv(features_path)
        logger.info(f"Loaded features: shape {self.X.shape}")

        # Load targets
        targets_path = Path(self.data_config['targets_path'])
        if not targets_path.exists():
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

        self.y = pd.read_csv(targets_path)

        # Handle target format based on multi-output configuration
        if self.is_multi_output:
            if self.y.shape[1] >= self.n_outputs:
                self.y = self.y.iloc[:, :self.n_outputs]  # Take first n_outputs columns
                logger.info(f"Multi-output: Using {self.n_outputs} target columns")
            else:
                raise ValueError(f"Expected {self.n_outputs} target columns, found {self.y.shape[1]}")
        else:
            # Single column target (original behavior)
            if self.y.shape[1] == 1:
                self.y = self.y.iloc[:, 0]
            else:
                logger.warning(f"Multiple target columns found: {self.y.shape[1]}. Using first column.")
                self.y = self.y.iloc[:, 0]

        logger.info(f"Loaded targets: shape {self.y.shape}")

        # Create sample weights - KEY NEW FUNCTIONALITY
        self.sample_weights = self.create_sample_weights(len(self.X))

        # Load feature types
        feature_types_path = Path(self.data_config['feature_types_path'])
        if feature_types_path.exists():
            with open(feature_types_path, 'rb') as f:
                self.feature_types = pickle.load(f)
            logger.info(f"Loaded feature types for {len(self.feature_types)} features")
        else:
            logger.warning("Feature types file not found. Treating all features as additive.")
            self.feature_types = {col: "0" for col in self.X.columns}

        # Store feature names
        self.feature_names = list(self.X.columns)

        # Validate data
        self._validate_data()

        return self.X, self.y, self.feature_types

    def split_data(self,
                   X: Optional[pd.DataFrame] = None,
                   y: Optional[Union[pd.Series, pd.DataFrame]] = None,
                   sample_weights: Optional[np.ndarray] = None,
                   test_size: Optional[float] = None,
                   stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame], np.ndarray, np.ndarray]:
        """
        Split data into train and test sets, preserving sample weights

        Args:
            X: Features DataFrame (uses self.X if None)
            y: Target Series/DataFrame (uses self.y if None)
            sample_weights: Sample weights array (uses self.sample_weights if None)
            test_size: Test set size (uses config value if None)
            stratify: Whether to stratify split (for classification)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, weights_train, weights_test)
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if sample_weights is None:
            sample_weights = self.sample_weights
        if test_size is None:
            test_size = self.data_config.get('test_size', 0.2)

        logger.info(f"Splitting data with test_size={test_size}")

        # No stratification for multi-output regression
        stratify_param = y if stratify and not self.is_multi_output else None

        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        logger.info(f"Weight distribution after split:")
        logger.info(f"- Train weights: sum={np.sum(weights_train):.0f}, mean={np.mean(weights_train):.3f}")
        logger.info(f"- Test weights: sum={np.sum(weights_test):.0f}, mean={np.mean(weights_test):.3f}")

        train_high_weight = np.sum(weights_train == 2.0)
        test_high_weight = np.sum(weights_test == 2.0)
        logger.info(f"- Train: {train_high_weight} high-weight samples out of {len(weights_train)}")
        logger.info(f"- Test: {test_high_weight} high-weight samples out of {len(weights_test)}")

        return X_train, X_test, y_train, y_test, weights_train, weights_test

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data

        Returns:
            Dictionary with data summary
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        feature_groups = self.get_feature_groups()

        # NEW: Handle both single and multi-output targets
        if isinstance(self.y, pd.DataFrame):
            summary = {
                'n_samples': len(self.X),
                'n_features': len(self.X.columns),
                'n_additive_features': len(feature_groups['additive']),
                'n_multiplicative_features': len(feature_groups['multiplicative']),
                'target_mean': self.y.mean().to_dict(),
                'target_std': self.y.std().to_dict(),
                'target_min': self.y.min().to_dict(),
                'target_max': self.y.max().to_dict(),
                'missing_features': int(self.X.isnull().sum().sum()),
                'missing_targets': int(self.y.isnull().sum().sum()),
                'n_targets': self.y.shape[1]
            }
        else:
            summary = {
                'n_samples': len(self.X),
                'n_features': len(self.X.columns),
                'n_additive_features': len(feature_groups['additive']),
                'n_multiplicative_features': len(feature_groups['multiplicative']),
                'target_mean': float(self.y.mean()),
                'target_std': float(self.y.std()),
                'target_min': float(self.y.min()),
                'target_max': float(self.y.max()),
                'missing_features': int(self.X.isnull().sum().sum()),
                'missing_targets': int(pd.isna(self.y).sum()),
                'n_targets': 1
            }

        # Add weight information - NEW SUMMARY STATS
        if self.sample_weights is not None:
            summary.update({
                'total_weight': float(np.sum(self.sample_weights)),
                'weight_mean': float(np.mean(self.sample_weights)),
                'weight_std': float(np.std(self.sample_weights)),
                'n_high_weight_samples': int(np.sum(self.sample_weights == 2.0)),
                'high_weight_percentage': float(np.sum(self.sample_weights == 2.0) / len(self.sample_weights) * 100)
            })

        return summary

    def get_feature_groups(self) -> Dict[str, list]:
        """
        Get feature groups by type

        Returns:
            Dictionary with 'additive' and 'multiplicative' feature lists
        """
        additive_features = [col for col, sign in self.feature_types.items() if sign == "0"]
        multiplicative_features = [col for col, sign in self.feature_types.items() if sign == "1"]

        return {
            'additive': additive_features,
            'multiplicative': multiplicative_features
        }

    def _validate_data(self):
        """Validate loaded data including sample weights"""
        # Check for matching sample sizes
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature and target sizes don't match: {len(self.X)} vs {len(self.y)}")

        # Check sample weights size - NEW VALIDATION
        if self.sample_weights is not None and len(self.sample_weights) != len(self.X):
            raise ValueError(f"Sample weights size doesn't match data size: {len(self.sample_weights)} vs {len(self.X)}")

        # Existing validation code...
        if self.X.isnull().any().any():
            n_missing = self.X.isnull().sum().sum()
            logger.warning(f"Found {n_missing} missing values in features")

        # Handle both DataFrame and Series for targets
        if isinstance(self.y, pd.DataFrame):
            if self.y.isnull().any().any():
                n_missing = self.y.isnull().sum().sum()
                logger.warning(f"Found {n_missing} missing values in targets")
        else:
            if pd.isna(self.y).any():
                n_missing = pd.isna(self.y).sum()
                logger.warning(f"Found {n_missing} missing values in targets")

        # Check feature types consistency
        missing_types = set(self.X.columns) - set(self.feature_types.keys())
        if missing_types:
            logger.warning(f"Missing feature types for {len(missing_types)} features. Setting to additive.")
            for col in missing_types:
                self.feature_types[col] = "0"

        # Validate feature type values
        invalid_types = [k for k, v in self.feature_types.items() if v not in ["0", "1"]]
        if invalid_types:
            raise ValueError(f"Invalid feature types found. Expected '0' or '1', got types for: {invalid_types[:5]}")

        logger.info("Data validation completed successfully")