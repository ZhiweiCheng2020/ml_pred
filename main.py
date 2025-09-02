"""
Main pipeline execution script
"""

import argparse
import logging
from pathlib import Path
import sys
import warnings

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
    logger.info("Starting Insurance ML Pipeline")
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
    logger.info("