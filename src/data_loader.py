"""
Data loading and splitting utilities
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
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

        self.X = None
        self.y = None
        self.feature_types = None
        self.feature_names = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """
        Load features, targets, and feature types

        Returns:
            Tuple of (features DataFrame, targets DataFrame, feature_types dict)
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

        # Handle single column target
        if self.y.shape[1] == 1:
            self.y = self.y.iloc[:, 0]
        else:
            logger.warning(f"Multiple target columns found: {self.y.shape[1]}. Using first column.")
            self.y = self.y.iloc[:, 0]

        logger.info(f"Loaded targets: shape {self.y.shape}")

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

    def _validate_data(self):
        """Validate loaded data"""
        # Check for matching sample sizes
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature and target sizes don't match: {len(self.X)} vs {len(self.y)}")

        # Check for missing values
        if self.X.isnull().any().any():
            n_missing = self.X.isnull().sum().sum()
            logger.warning(f"Found {n_missing} missing values in features")

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

    def split_data(self,
                   X: Optional[pd.DataFrame] = None,
                   y: Optional[pd.Series] = None,
                   test_size: Optional[float] = None,
                   stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets

        Args:
            X: Features DataFrame (uses self.X if None)
            y: Target Series (uses self.y if None)
            test_size: Test set size (uses config value if None)
            stratify: Whether to stratify split (for classification)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if test_size is None:
            test_size = self.data_config.get('test_size', 0.2)

        logger.info(f"Splitting data with test_size={test_size}")

        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

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

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data

        Returns:
            Dictionary with data summary
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        feature_groups = self.get_feature_groups()

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
            'missing_targets': int(pd.isna(self.y).sum())
        }

        return summary