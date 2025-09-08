"""
Neural network models - Fixed dimension handling with proper SAINT implementation
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import warnings
import math

warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers, optimizers

# PyTorch imports for TabNet and SAINT
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetRegressor

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SparseNNModel(BaseModel):
    """Sparse Neural Network with L1 regularization"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Sparse NN model"""
        super().__init__(config)
        self.model_name = 'sparse_nn'
        self.architecture = config.get('architecture', {})
        self.training_params = config.get('training', {})
        self.regularization = config.get('regularization', {})

    def build_model(self, input_dim: int = None) -> keras.Model:
        """Build Sparse NN model"""
        if input_dim is None:
            input_dim = self.architecture.get('input_dim', 100)

        # Build model
        model = models.Sequential()

        # Input layer
        model.add(layers.InputLayer(input_shape=(input_dim,)))

        # Hidden layers with L1 regularization
        hidden_layers_config = self.architecture.get('hidden_layers', [])
        for layer_config in hidden_layers_config:
            units = layer_config.get('units', 128)
            activation = layer_config.get('activation', 'relu')
            dropout_rate = layer_config.get('dropout', 0.2)
            l1_reg = layer_config.get('l1_regularization', 0.01)

            # Dense layer with L1 regularization
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l1(l1_reg)
            ))

            # Batch normalization
            if self.regularization.get('batch_norm', True):
                model.add(layers.BatchNormalization())

            # Dropout
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        # Output layer
        output_dim = self.architecture.get('output_dim', 1)
        output_activation = self.architecture.get('output_activation', 'linear')
        model.add(layers.Dense(output_dim, activation=output_activation))

        # Compile model
        learning_rate = self.training_params.get('learning_rate', 0.001)
        optimizer_type = self.training_params.get('optimizer', 'adam')

        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        loss = self.training_params.get('loss', 'mse')
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'SparseNNModel':
        """Fit Sparse NN model"""
        logger.info(f"Training {self.model_name} model...")

        # Ensure y is properly shaped (1D for regression)
        y = self._ensure_1d_target(y)
        if y_val is not None:
            y_val = self._ensure_1d_target(y_val)

        # Build model
        input_dim = X.shape[1]
        self.model = self.build_model(input_dim)

        # Prepare callbacks
        callbacks_list = []

        # Early stopping
        early_stopping_config = self.training_params.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            callbacks_list.append(callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_config.get('patience', 20),
                min_delta=early_stopping_config.get('min_delta', 0.0001),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            ))

        # Reduce learning rate
        reduce_lr_config = self.training_params.get('reduce_lr', {})
        if reduce_lr_config.get('enabled', True):
            callbacks_list.append(callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=reduce_lr_config.get('factor', 0.5),
                patience=reduce_lr_config.get('patience', 10),
                min_lr=reduce_lr_config.get('min_lr', 1e-5)
            ))

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model
        batch_size = self.training_params.get('batch_size', 32)
        epochs = self.training_params.get('epochs', 200)
        validation_split = self.training_params.get('validation_split', 0.2) if validation_data is None else 0.0

        self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=0
        )

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def _ensure_1d_target(self, y: np.ndarray) -> np.ndarray:
        """Ensure target is 1D for regression"""
        if len(y.shape) > 1:
            if y.shape[1] == 1:
                return y.flatten()
            else:
                logger.warning(f"Target has multiple columns: {y.shape}. Using first column.")
                return y[:, 0]
        return y


class TabNetModel(BaseModel):
    """TabNet model"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize TabNet model"""
        super().__init__(config)
        self.model_name = 'tabnet'
        self.architecture = config.get('architecture', {})
        self.training_params = config.get('training', {})
        self.regularization = config.get('regularization', {})

    def build_model(self) -> TabNetRegressor:
        """Build TabNet model"""
        # Extract parameters
        params = {
            'n_d': self.architecture.get('n_d', 8),
            'n_a': self.architecture.get('n_a', 8),
            'n_steps': self.architecture.get('n_steps', 3),
            'gamma': self.architecture.get('gamma', 1.3),
            'n_independent': self.architecture.get('n_independent', 2),
            'n_shared': self.architecture.get('n_shared', 2),
            'lambda_sparse': self.regularization.get('lambda_sparse', 0.001),
            'momentum': self.architecture.get('momentum', 0.02),
            'clip_value': self.training_params.get('clip_value', 1.0),
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': self.training_params.get('learning_rate', 0.02)},
            'scheduler_fn': torch.optim.lr_scheduler.CosineAnnealingLR,
            'scheduler_params': {'T_max': 50},
            'mask_type': self.architecture.get('mask_type', 'sparsemax'),
            'verbose': 0,
            'seed': 42
        }

        return TabNetRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'TabNetModel':
        """Fit TabNet model"""
        logger.info(f"Training {self.model_name} model...")

        # Build model
        self.model = self.build_model()

        # Prepare training parameters
        max_epochs = self.training_params.get('epochs', 200)
        batch_size = self.training_params.get('batch_size', 1024)
        virtual_batch_size = self.training_params.get('virtual_batch_size', 128)
        patience = self.training_params.get('early_stopping', {}).get('patience', 30)

        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # Ensure targets are 2D for TabNet (it expects 2D targets)
        y_train_2d = self._ensure_2d_target(y)
        y_val_2d = None
        if y_val is not None:
            y_val_2d = self._ensure_2d_target(y_val)
            eval_set = [(X_val, y_val_2d)]

        # Train model
        try:
            self.model.fit(
                X, y_train_2d,
                eval_set=eval_set,
                max_epochs=max_epochs,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                patience=patience,
                eval_metric=['rmse']
            )
        except Exception as e:
            logger.error(f"TabNet training failed: {e}")
            # Try with simpler parameters
            logger.info("Retrying with simpler TabNet configuration...")
            self.model = TabNetRegressor(
                n_d=8, n_a=8, n_steps=3, gamma=1.3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={'lr': 0.02},
                verbose=0, seed=42
            )
            self.model.fit(
                X, y_train_2d,
                eval_set=eval_set,
                max_epochs=min(100, max_epochs),
                batch_size=min(512, batch_size),
                virtual_batch_size=min(64, virtual_batch_size),
                patience=min(15, patience)
            )

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X)
        return predictions.flatten()

    def _ensure_2d_target(self, y: np.ndarray) -> np.ndarray:
        """Ensure target is 2D for TabNet"""
        if len(y.shape) == 1:
            return y.reshape(-1, 1)
        return y


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear layer
        output = self.W_o(attn_output)

        # Residual connection and layer norm
        return self.layer_norm(output + query)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)


class IntersampleAttention(nn.Module):
    """Intersample attention mechanism for SAINT"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, intersample_x):
        """
        Args:
            x: Current sample embeddings [batch_size, seq_len, d_model]
            intersample_x: Other samples for intersample attention [batch_size, seq_len, d_model]
        """
        return self.attention(x, intersample_x, intersample_x)


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)


class SAINTTransformerBlock(nn.Module):
    """SAINT Transformer block with self-attention and intersample attention"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 use_intersample: bool = True):
        super().__init__()
        self.use_intersample = use_intersample

        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Intersample attention (if enabled)
        if use_intersample:
            self.intersample_attention = IntersampleAttention(d_model, num_heads, dropout)

        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, intersample_x=None):
        # Self-attention
        x = self.self_attention(x, x, x)

        # Intersample attention
        if self.use_intersample and intersample_x is not None:
            x = self.intersample_attention(x, intersample_x)

        # Feed-forward
        x = self.feed_forward(x)

        return x


class SAINTCore(nn.Module):
    """Core SAINT model architecture"""

    def __init__(self,
                 input_dim: int,
                 d_model: int = 32,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = None,
                 dropout: float = 0.1,
                 use_intersample: bool = True,
                 output_dim: int = 1):
        super().__init__()

        self.d_model = d_model
        self.use_intersample = use_intersample

        if d_ff is None:
            d_ff = d_model * 4