"""
Model Architectures for Production Inference
=============================================

Contains all model architectures used in production.
The ModelRegistry imports these to reconstruct models from state dicts.

Architecture naming convention:
- ProductionLSTM: Simple LSTM with attention (2-class)
- ImprovedLSTM: Bidirectional LSTM with multi-head attention (3-class)
- TransformerPredictor: Transformer-based model
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head self-attention for temporal sequences."""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output, attn_weights


class ProductionLSTM(nn.Module):
    """
    Production LSTM model architecture (binary classification).

    Architecture:
    - LSTM layers with attention
    - Binary output (up/down)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output


class ImprovedLSTM(nn.Module):
    """
    Improved LSTM with bidirectional layers and multi-head attention (3-class).

    Architecture:
    - Input projection with LayerNorm
    - Bidirectional LSTM
    - Multi-head temporal attention with residual
    - 3-class output (down/neutral/up)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_classes: int = 3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention
        self.attention = MultiHeadTemporalAttention(
            input_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout * 0.5
        )
        self.attention_norm = nn.LayerNorm(hidden_size * 2)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        self._init_weights()

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        context = attn_out.mean(dim=1)
        output = self.fc(context)
        return output

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class SimpleLSTM(nn.Module):
    """
    Simple LSTM for basic predictions.
    Used by optimized/aggressive LSTM variants.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output


# Architecture registry mapping model_type to class
ARCHITECTURE_REGISTRY: Dict[str, type] = {
    "production_lstm": ProductionLSTM,
    "improved_lstm": ImprovedLSTM,
    "simple_lstm": SimpleLSTM,
    "lstm": ImprovedLSTM,  # Default LSTM maps to ImprovedLSTM
}


def get_architecture(
    model_type: str,
    architecture_config: Dict[str, Any]
) -> nn.Module:
    """
    Get model architecture from type and config.

    Args:
        model_type: Type of model (lstm, transformer, etc.)
        architecture_config: Dict with architecture parameters

    Returns:
        Instantiated nn.Module (not loaded with weights)
    """
    # Normalize model type
    model_type_lower = model_type.lower()

    # Get architecture class
    if model_type_lower in ARCHITECTURE_REGISTRY:
        arch_class = ARCHITECTURE_REGISTRY[model_type_lower]
    elif "lstm" in model_type_lower:
        arch_class = ImprovedLSTM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Extract parameters with defaults
    input_dim = architecture_config.get("input_dim", architecture_config.get("input_size", 113))
    hidden_size = architecture_config.get("hidden_size", 256)
    num_layers = architecture_config.get("num_layers", 3)
    dropout = architecture_config.get("dropout", 0.3)
    num_classes = architecture_config.get("num_classes", 3)

    # Instantiate
    if arch_class == ProductionLSTM:
        return arch_class(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif arch_class == SimpleLSTM:
        return arch_class(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes
        )
    else:
        return arch_class(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes
        )


def infer_architecture_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer architecture config from state dict keys and shapes.

    Args:
        state_dict: Model state dictionary

    Returns:
        Inferred architecture config
    """
    config = {}

    # Try to infer input_dim from first layer
    for key, tensor in state_dict.items():
        if "input_proj.0.weight" in key:
            # ImprovedLSTM: input_proj is Linear(input_dim, hidden_size)
            config["input_dim"] = tensor.shape[1]
            config["hidden_size"] = tensor.shape[0]
            config["architecture"] = "improved_lstm"
            break
        elif "lstm.weight_ih_l0" in key:
            # Standard LSTM: weight_ih_l0 shape is (4*hidden_size, input_dim)
            config["hidden_size"] = tensor.shape[0] // 4
            config["input_dim"] = tensor.shape[1]
            break

    # Count LSTM layers
    lstm_layers = set()
    for key in state_dict.keys():
        if "lstm.weight_ih_l" in key:
            layer_num = int(key.split("lstm.weight_ih_l")[1].split("_")[0])
            lstm_layers.add(layer_num)
    config["num_layers"] = len(lstm_layers) if lstm_layers else 3

    # Check if bidirectional
    config["bidirectional"] = any("_reverse" in k for k in state_dict.keys())

    # Infer num_classes from output layer
    for key, tensor in state_dict.items():
        if "fc" in key and "weight" in key:
            # Last fc layer determines num_classes
            pass  # Will be last one we see

    # Find the final output dimension
    fc_weights = [(k, v) for k, v in state_dict.items() if "fc" in k and "weight" in k]
    if fc_weights:
        last_fc_key, last_fc_tensor = fc_weights[-1]
        config["num_classes"] = last_fc_tensor.shape[0]

    # Default dropout
    config["dropout"] = 0.3

    return config


__all__ = [
    "ProductionLSTM",
    "ImprovedLSTM",
    "SimpleLSTM",
    "MultiHeadTemporalAttention",
    "ARCHITECTURE_REGISTRY",
    "get_architecture",
    "infer_architecture_config",
]
