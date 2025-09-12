"""
Neural network models - Minimal changes for multi-output support
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import warnings
import math
import os

warnings.filterwarnings('ignore')

# TensorFlow imports (2.15+)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers, optimizers

# PyTorch imports for TabNet and SAINT (2.1+)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetRegressor

from .base_model import BaseModel

logger = logging.getLogger(__name__)


def setup_gpu_tensorflow(config: Dict[str, Any]) -> None:
    """Setup GPU for TensorFlow 2.15+"""
    gpu_enabled = config.get('resources', {}).get('gpu_enabled', False)

    if gpu_enabled:
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow GPU enabled: {len(gpus)} GPU(s) available")

                # Mixed precision for TF 2.15+
                # Use legacy keras for compatibility
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training enabled for TensorFlow")
            except RuntimeError as e:
                logger.error(f"GPU setup error: {e}")
        else:
            logger.warning("GPU enabled in config but no GPU detected for TensorFlow")
    else:
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("TensorFlow GPU disabled")


def setup_gpu_pytorch(config: Dict[str, Any]) -> torch.device:
    """Setup GPU for PyTorch 2.1+"""
    gpu_enabled = config.get('resources', {}).get('gpu_enabled', False)

    if gpu_enabled and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"PyTorch GPU enabled: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True

        # For PyTorch 2.1+, use the new default tensor type setting
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device('cuda')
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        if gpu_enabled:
            logger.warning("GPU enabled in config but CUDA not available for PyTorch")
        else:
            logger.info("PyTorch GPU disabled")

    return device


class SparseNNModel(BaseModel):
    """Sparse Neural Network with L1 regularization - TF 2.15+ compatible"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Sparse NN model"""
        super().__init__(config)
        self.model_name = 'sparse_nn'
        self.architecture = config.get('architecture', {})
        self.training_params = config.get('training', {})
        self.regularization = config.get('regularization', {})

        # Setup GPU for TensorFlow
        setup_gpu_tensorflow(config)

    def build_model(self, input_dim: int = None) -> keras.Model:
        """Build Sparse NN model with GPU optimization"""
        if input_dim is None:
            input_dim = self.architecture.get('input_dim', 100)

        # Check if GPU is available
        gpu_enabled = self.config.get('resources', {}).get('gpu_enabled', False)
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0

        # Build model
        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

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
                kernel_regularizer=regularizers.L1(l1_reg)  # Updated for latest TF
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

        # For mixed precision, ensure output is float32
        if gpu_enabled and gpu_available:
            model.add(layers.Dense(output_dim, activation=output_activation, dtype='float32'))
        else:
            model.add(layers.Dense(output_dim, activation=output_activation))

        # Compile model with updated optimizer syntax
        learning_rate = self.training_params.get('learning_rate', 0.001)
        optimizer_type = self.training_params.get('optimizer', 'adam')

        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        # For mixed precision with TF 2.15+
        if gpu_enabled and gpu_available:
            from tensorflow.keras import mixed_precision
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        loss = self.training_params.get('loss', 'mse')
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'SparseNNModel':
        """Fit Sparse NN model with GPU optimization"""
        logger.info(f"Training {self.model_name} model...")

        # Check GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_available:
            logger.info(f"Training on GPU: {tf.config.list_physical_devices('GPU')}")

        y = np.array(y)
        if y_val is not None:
            y_val = np.array(y_val)

        # Ensure data is float32 to avoid type conflicts
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
        if y_val is not None:
            y_val = y_val.astype(np.float32)

        # Build model
        input_dim = X.shape[1]
        self.model = self.build_model(input_dim)

        # Prepare callbacks with updated syntax for TF 2.15+
        callbacks_list = []

        # Early stopping
        early_stopping_config = self.training_params.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            callbacks_list.append(callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_config.get('patience', 20),
                min_delta=early_stopping_config.get('min_delta', 0.0001),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True),
                verbose=0
            ))

        # Reduce learning rate
        reduce_lr_config = self.training_params.get('reduce_lr', {})
        if reduce_lr_config.get('enabled', True):
            callbacks_list.append(callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=reduce_lr_config.get('factor', 0.5),
                patience=reduce_lr_config.get('patience', 10),
                min_lr=reduce_lr_config.get('min_lr', 1e-5),
                verbose=0
            ))

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model with optimized batch size for GPU
        batch_size = self.training_params.get('batch_size', 32)
        if gpu_available:
            batch_size = max(batch_size, 256)

        epochs = self.training_params.get('epochs', 200)
        validation_split = self.training_params.get('validation_split', 0.2) if validation_data is None else 0.0

        # Train with standard fit method (simpler and more stable)
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
        """Make predictions with GPU optimization"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        batch_size = 1024 if gpu_available else 32

        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return predictions

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
    """TabNet model with GPU support - Compatible with pytorch-tabnet 4.1+"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize TabNet model"""
        super().__init__(config)
        self.model_name = 'tabnet'
        self.architecture = config.get('architecture', {})
        self.training_params = config.get('training', {})
        self.regularization = config.get('regularization', {})

        # Setup GPU for PyTorch
        self.device = setup_gpu_pytorch(config)

    def build_model(self) -> TabNetRegressor:
        """Build TabNet model with GPU support"""
        # Extract parameters - updated for pytorch-tabnet 4.1+
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
            'seed': 42,
            'device_name': 'cuda' if self.device.type == 'cuda' else 'cpu'
        }

        return TabNetRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'TabNetModel':
        """Fit TabNet model with GPU optimization"""
        logger.info(f"Training {self.model_name} model on {self.device}...")

        # Build model
        self.model = self.build_model()

        # Prepare training parameters - updated for pytorch-tabnet 4.1+
        max_epochs = self.training_params.get('epochs', 200)
        batch_size = self.training_params.get('batch_size', 1024)

        # Increase batch size for GPU
        if self.device.type == 'cuda':
            batch_size = max(batch_size, 2048)

        virtual_batch_size = self.training_params.get('virtual_batch_size', 128)
        patience = self.training_params.get('early_stopping', {}).get('patience', 30)

        # Ensure targets are 2D for TabNet
        y_train_2d = self._ensure_2d_target(y)

        # Prepare eval set
        eval_set = []
        if X_val is not None and y_val is not None:
            y_val_2d = self._ensure_2d_target(y_val)
            eval_set = [(X_val, y_val_2d)]

        # Train model with updated parameters for pytorch-tabnet 4.1+
        try:
            self.model.fit(
                X, y_train_2d,
                eval_set=eval_set if eval_set else None,
                max_epochs=max_epochs,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                patience=patience,
                eval_metric=['rmse'],
                drop_last=False,  # New parameter in recent versions
                augmentations=None  # Explicitly set to None
            )
        except Exception as e:
            logger.error(f"TabNet training failed: {e}")
            logger.info("Retrying with simpler TabNet configuration...")

            # Fallback configuration
            self.model = TabNetRegressor(
                n_d=8, n_a=8, n_steps=3, gamma=1.3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={'lr': 0.02},
                verbose=0, seed=42,
                device_name='cuda' if self.device.type == 'cuda' else 'cpu'
            )
            self.model.fit(
                X, y_train_2d,
                eval_set=eval_set if eval_set else None,
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
        return predictions

    def _ensure_2d_target(self, y: np.ndarray) -> np.ndarray:
        """Ensure target is 2D for TabNet"""
        if len(y.shape) == 1:
            return y.reshape(-1, 1)
        return y


# SAINT Model components with GPU support for PyTorch 2.1+
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with PyTorch 2.1+ optimizations"""

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

        # Use PyTorch 2.0+ scaled_dot_product_attention for better performance
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,  # Enable Flash Attention
            enable_math=True,
            enable_mem_efficient=True
        ):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear layer
        output = self.W_o(attn_output)

        # Residual connection and layer norm
        return self.layer_norm(output + query)


class SAINTModel(BaseModel):
    """SAINT model with GPU support - PyTorch 2.1+ compatible"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SAINT model"""
        super().__init__(config)
        self.model_name = 'saint'
        self.architecture = config.get('architecture', {})
        self.training_params = config.get('training', {})

        # Setup GPU for PyTorch
        self.device = setup_gpu_pytorch(config)

    def build_model(self, input_dim: int) -> nn.Module:
        """Build SAINT model - simplified version"""
        logger.warning("SAINT model using simplified implementation")

        class SimpleSAINT(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, output_dim=1):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.1)
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                x = F.relu(self.norm1(self.fc1(x)))
                x = self.dropout(x)
                x = F.relu(self.norm2(self.fc2(x)))
                x = self.dropout(x)
                return self.fc3(x)

        output_dim = self.architecture.get('output_dim', 1)

        model = SimpleSAINT(
            input_dim,
            hidden_dim=self.architecture.get('dim', 64),
            output_dim=output_dim
        )

        return model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'SAINTModel':
        """Fit SAINT model with GPU optimization"""
        logger.info(f"Training {self.model_name} model on {self.device}...")

        # Build model
        input_dim = X.shape[1]
        self.model = self.build_model(input_dim)

        y = np.array(y)
        if y_val is not None:
            y_val = np.array(y_val)

        # Convert to tensors with proper dtype
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Setup optimizer with PyTorch 2.1+ features
        learning_rate = self.training_params.get('learning_rate', 0.001)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_params.get('weight_decay', 0.01),
            fused=True if self.device.type == 'cuda' else False  # Fused optimizer for GPU
        )

        # Enable torch.compile for PyTorch 2.0+ (optional, can improve performance)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Training parameters
        batch_size = self.training_params.get('batch_size', 256)
        if self.device.type == 'cuda':
            batch_size = max(batch_size, 512)

        epochs = self.training_params.get('epochs', 150)

        # Simple training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)

            if len(y_tensor.shape) > 1 and y_tensor.shape[1] > 1:
                # Multi-output
                loss = F.mse_loss(outputs, y_tensor)
            else:
                # Single output
                loss = F.mse_loss(outputs.squeeze(), y_tensor.flatten())

            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        logger.info(f"{self.model_name} model trained successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions