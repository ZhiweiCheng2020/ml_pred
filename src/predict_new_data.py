
import sys
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from pathlib import Path
import joblib
import json
import warnings
from typing import Dict, Any, Optional, List, Tuple

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.utils import load_config
from src.preprocessor import Preprocessor
from src.feature_selector import FeatureSelector
from src.models import get_model

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handle predictions on new data using trained models"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize predictor

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.models_to_run = self.config.get('models_to_run', [])
        self.results_dir = Path(self.config.get('output', {}).get('results_dir', 'results'))
        self.model_artifacts_dir = self.results_dir / 'model_artifacts'

        # Storage for loaded components
        self.trained_models = {}
        self.preprocessors = {}
        self.feature_selectors = {}
        self.model_configs = {}
        self.feature_types = None

    def load_feature_types(self) -> Dict[str, str]:
        """Load feature types from pickle file"""
        feature_types_path = Path(self.config['data']['feature_types_path'])

        if feature_types_path.exists():
            with open(feature_types_path, 'rb') as f:
                feature_types = pickle.load(f)
            logger.info(f"Loaded feature types for {len(feature_types)} features")
            return feature_types
        else:
            logger.warning("Feature types file not found. Treating all features as additive.")
            return {}

    def load_new_data(self, data_path: str) -> pd.DataFrame:
        """
        Load new data for prediction

        Args:
            data_path: Path to new data CSV file

        Returns:
            DataFrame with features
        """
        logger.info(f"Loading new data from {data_path}")

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        X_new = pd.read_csv(data_path)
        logger.info(f"Loaded new data: shape {X_new.shape}")

        # Validate features match training data
        if self.feature_types:
            expected_features = set(self.feature_types.keys())
            actual_features = set(X_new.columns)

            missing_features = expected_features - actual_features
            extra_features = actual_features - expected_features

            if missing_features:
                logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X_new[feature] = 0

            if extra_features:
                logger.warning(f"Extra features in new data (will be dropped): {extra_features}")
                # Drop extra features
                X_new = X_new[list(expected_features)]

        return X_new

    def load_trained_model(self, model_name: str) -> bool:
        """
        Load a trained model and its components

        Args:
            model_name: Name of the model to load

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading {model_name} model and components...")

            # Load model configuration
            model_config_path = Path(f"config/models/{model_name}.yaml")
            if model_config_path.exists():
                self.model_configs[model_name] = load_config(str(model_config_path))
            else:
                logger.warning(f"Model config not found: {model_config_path}")
                return False

            # Load saved model artifacts
            model_artifact_path = self.model_artifacts_dir / f"{model_name}_model.pkl"
            if model_artifact_path.exists():
                with open(model_artifact_path, 'rb') as f:
                    model_data = joblib.load(f)
                    self.trained_models[model_name] = model_data
                logger.info(f"Loaded {model_name} model from {model_artifact_path}")
            else:
                # Try to load from results directory
                model_results_path = self.results_dir / 'models' / f"{model_name}_results.json"
                if model_results_path.exists():
                    logger.info(f"Model artifact not found, checking results: {model_results_path}")
                    # Model might not be saved separately, will need to retrain
                    logger.warning(f"Model {model_name} not found in artifacts. May need to retrain.")
                    return False
                else:
                    logger.error(f"No saved model found for {model_name}")
                    return False

            # Load preprocessor if exists
            preprocessor_path = self.model_artifacts_dir / f"{model_name}_preprocessor.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessors[model_name] = joblib.load(f)
                logger.info(f"Loaded preprocessor for {model_name}")

            # Load feature selector if exists
            selector_path = self.model_artifacts_dir / f"{model_name}_feature_selector.pkl"
            if selector_path.exists():
                with open(selector_path, 'rb') as f:
                    self.feature_selectors[model_name] = joblib.load(f)
                logger.info(f"Loaded feature selector for {model_name}")

            return True

        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return False

    def predict_single_model(self, model_name: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using a single model

        Args:
            model_name: Name of the model
            X: Features DataFrame

        Returns:
            Predictions array or None if failed
        """
        try:
            logger.info(f"Making predictions with {model_name}...")

            # Get the model
            if model_name not in self.trained_models:
                logger.error(f"Model {model_name} not loaded")
                return None

            model = self.trained_models[model_name]

            # Apply preprocessing if exists
            X_processed = X.copy()
            if model_name in self.preprocessors:
                preprocessor = self.preprocessors[model_name]
                X_processed = preprocessor.transform(X_processed)
                logger.info(f"Applied preprocessing for {model_name}")

            # Apply feature selection if exists
            if model_name in self.feature_selectors:
                selector = self.feature_selectors[model_name]
                X_processed = selector.transform(X_processed)
                logger.info(f"Applied feature selection for {model_name}")

            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X_processed)
            elif isinstance(model, dict) and 'model' in model:
                # Handle case where model is stored in a dictionary
                predictions = model['model'].predict(X_processed)
            else:
                logger.error(f"Model {model_name} does not have predict method")
                return None

            logger.info(f"Generated {len(predictions)} predictions for {model_name}")

            return predictions

        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def save_model_artifacts(self):
        """Save model artifacts after training (to be called from main pipeline)"""
        logger.info("Saving model artifacts for future predictions...")

        self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # This method should be called from the main training pipeline
        # to save models after training
        pass

    def predict_all_models(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using all configured models

        Args:
            X: Features DataFrame

        Returns:
            Dictionary of model predictions
        """
        predictions = {}

        for model_name in self.models_to_run:
            if model_name == 'ensemble':
                # Ensemble requires base models to be loaded first
                logger.info("Skipping ensemble model (requires special handling)")
                continue

            logger.info(f"\nProcessing {model_name}...")

            # Load model if not already loaded
            if model_name not in self.trained_models:
                success = self.load_trained_model(model_name)
                if not success:
                    logger.warning(f"Skipping {model_name} - could not load model")
                    continue

            # Make predictions
            model_predictions = self.predict_single_model(model_name, X)

            if model_predictions is not None:
                predictions[model_name] = model_predictions
            else:
                logger.warning(f"Failed to generate predictions for {model_name}")

        return predictions

    def save_predictions_to_excel(self, predictions: Dict[str, np.ndarray],
                                  output_path: str,
                                  X: pd.DataFrame,
                                  include_features: bool = False) -> None:
        """
        Save predictions to Excel file with one sheet per model

        Args:
            predictions: Dictionary of model predictions
            output_path: Path for output Excel file
            X: Original features DataFrame (for reference)
            include_features: Whether to include input features in output
        """
        logger.info(f"Saving predictions to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Save summary sheet
            summary_data = []
            for model_name, preds in predictions.items():
                summary_data.append({
                    'Model': model_name,
                    'Num_Predictions': len(preds),
                    'Mean': np.mean(preds),
                    'Std': np.std(preds),
                    'Min': np.min(preds),
                    'Max': np.max(preds),
                    'Median': np.median(preds)
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Save predictions for each model
            for model_name, preds in predictions.items():
                # Create DataFrame for this model
                model_df = pd.DataFrame()

                # Add index
                model_df['Index'] = range(len(preds))

                # Add predictions
                model_df[f'{model_name}_Prediction'] = preds

                # Optionally add selected features for reference
                if include_features and len(X) == len(preds):
                    # Add first few features for reference (to avoid huge Excel files)
                    n_features_to_include = min(10, len(X.columns))
                    for col in X.columns[:n_features_to_include]:
                        model_df[col] = X[col].values

                # Save to sheet (Excel sheet names limited to 31 characters)
                sheet_name = model_name[:31] if len(model_name) > 31 else model_name
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)

                logger.info(f"Saved {model_name} predictions to sheet '{sheet_name}'")

            # Add metadata sheet
            metadata = {
                'Property': ['Prediction Date', 'Number of Samples', 'Number of Features',
                             'Number of Models', 'Config File'],
                'Value': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                          len(X), len(X.columns), len(predictions), 'config/config.yaml']
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        logger.info(f"Successfully saved predictions to {output_path}")

        # Also save as CSV for each model (easier to work with programmatically)
        csv_dir = output_path.parent / 'predictions_csv'
        csv_dir.mkdir(parents=True, exist_ok=True)

        for model_name, preds in predictions.items():
            csv_path = csv_dir / f"{model_name}_predictions.csv"
            pred_df = pd.DataFrame({
                'prediction': preds
            })
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"Also saved {model_name} predictions to {csv_path}")


def main():
    """Main function for prediction on new data"""
    parser = argparse.ArgumentParser(description="Predict on new data using trained models")
    parser.add_argument("--data", required=True, help="Path to new data CSV file")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", default="predictions/predictions.xlsx",
                        help="Path for output Excel file")
    parser.add_argument("--include-features", action="store_true",
                        help="Include input features in output Excel")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to use (overrides config)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("PREDICTION ON NEW DATA")
    logger.info("=" * 60)

    try:
        # Initialize predictor
        predictor = ModelPredictor(args.config)

        # Override models if specified
        if args.models:
            predictor.models_to_run = args.models
            logger.info(f"Using specified models: {args.models}")
        else:
            logger.info(f"Using models from config: {predictor.models_to_run}")

        # Load feature types
        predictor.feature_types = predictor.load_feature_types()

        # Load new data
        X_new = predictor.load_new_data(args.data)

        logger.info("\n" + "=" * 60)
        logger.info("GENERATING PREDICTIONS")
        logger.info("=" * 60)

        # Make predictions
        predictions = predictor.predict_all_models(X_new)

        if not predictions:
            logger.error("No predictions generated. Please check if models are trained and saved.")
            logger.info("\nTo use this script, first run the training pipeline:")
            logger.info("  python main.py --config config/config.yaml")
            logger.info("\nThen ensure model artifacts are saved in: results/model_artifacts/")
            return

        logger.info("\n" + "=" * 60)
        logger.info("SAVING RESULTS")
        logger.info("=" * 60)

        # Save predictions to Excel
        predictor.save_predictions_to_excel(
            predictions,
            args.output,
            X_new,
            include_features=args.include_features
        )

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)

        for model_name, preds in predictions.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Predictions: {len(preds)}")
            logger.info(f"  Mean: {np.mean(preds):.6f}")
            logger.info(f"  Std: {np.std(preds):.6f}")
            logger.info(f"  Min: {np.min(preds):.6f}")
            logger.info(f"  Max: {np.max(preds):.6f}")

        logger.info("\n" + "=" * 60)
        logger.info(f"COMPLETE - Predictions saved to {args.output}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()