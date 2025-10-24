"""
Transformer models for waveform classification

Includes:
    - Standard Transformer
    - Vision Transformer (ViT) adapted for 1D waveforms
"""

import torch
import torch.nn as nn
import math
from .models import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information"""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WaveformTransformer(BaseModel):
    """
    Standard Transformer for 1D waveform classification

    Uses self-attention to learn important temporal features
    """

    def __init__(
        self,
        waveform_length: int = 1024,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256
    ):
        super().__init__(num_classes=num_classes)

        self.d_model = d_model

        # Input projection (1D signal to d_model dimensions)
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, waveform_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Forward pass

        Parameters:
            x: Input waveforms, shape (batch, samples)
            return_attention: If True, return attention weights (not implemented in this version)

        Returns:
            Logits, shape (batch, num_classes)
        """
        # x: [batch, 1024]
        batch_size = x.shape[0]

        # Reshape for projection
        x = x.unsqueeze(-1)  # [batch, 1024, 1]

        # Project to d_model
        x = self.input_proj(x)  # [batch, 1024, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # [batch, 1024, d_model]

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # [batch, d_model]

        # Classification
        logits = self.classifier(x)  # [batch, num_classes]

        return logits


class VisionTransformerWaveform(BaseModel):
    """
    Vision Transformer adapted for 1D waveforms

    Splits waveform into patches and processes with transformer
    """

    def __init__(
        self,
        waveform_length: int = 1024,
        patch_size: int = 16,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(num_classes=num_classes)

        assert waveform_length % patch_size == 0, "waveform_length must be divisible by patch_size"

        self.patch_size = patch_size
        self.num_patches = waveform_length // patch_size
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Parameters:
            x: Input waveforms, shape (batch, samples)

        Returns:
            Logits, shape (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Create patches: [batch, num_patches, patch_size]
        x = x.view(batch_size, self.num_patches, self.patch_size)

        # Embed patches: [batch, num_patches, d_model]
        x = self.patch_embed(x)

        # Add CLS token: [batch, num_patches + 1, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]  # [batch, d_model]

        # Classification
        logits = self.classifier(cls_output)

        return logits


class TransformerWithAttentionMaps(BaseModel):
    """
    Transformer that returns attention maps for interpretability

    Based on WaveformTransformer but extracts attention weights
    """

    def __init__(
        self,
        waveform_length: int = 1024,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(num_classes=num_classes)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, waveform_length, dropout)

        # Custom transformer encoder layers (to extract attention)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Forward pass with optional attention map extraction

        Parameters:
            x: Input waveforms, shape (batch, samples)
            return_attention: If True, return attention weights

        Returns:
            If return_attention=False: logits, shape (batch, num_classes)
            If return_attention=True: (logits, attention_maps)
        """
        # Input projection
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Apply transformer layers
        attention_maps = []

        for layer in self.encoder_layers:
            if return_attention:
                # Note: PyTorch TransformerEncoderLayer doesn't expose attention weights by default
                # This is a simplified version - for full attention extraction, you'd need
                # to implement custom MultiheadAttention
                x = layer(x)
                attention_maps.append(None)  # Placeholder
            else:
                x = layer(x)

        # Global pooling
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        if return_attention:
            return logits, attention_maps
        else:
            return logits
