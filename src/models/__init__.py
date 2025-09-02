"""
Models package
"""

from .base_model import BaseModel
from .linear_models import RidgeModel, ElasticNetModel
from .tree_models import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel
from .neural_models import SparseNNModel, TabNetModel, SAINTModel
from .ensemble_models import EnsembleModel

__all__ = [
    'BaseModel',
    'RidgeModel',
    'ElasticNetModel',
    'RandomForestModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'SparseNNModel',
    'TabNetModel',
    'SAINTModel',
    'EnsembleModel'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'ridge': RidgeModel,
    'elasticnet': ElasticNetModel,
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'sparse_nn': SparseNNModel,
    'tabnet': TabNetModel,
    'saint': SAINTModel,
    'ensemble': EnsembleModel
}


def get_model(model_name: str, config: dict):
    """
    Get model instance by name

    Args:
        model_name: Name of the model
        config: Model configuration

    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(config)