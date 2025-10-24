"""
Hybrid architectures combining CNNs and Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import BaseModel


class CNNTransformerHybrid(BaseModel):
    """
    Hybrid model: CNN for local features + Transformer for global context

    Architecture:
        1. CNN backbone extracts local features
        2. Transformer processes feature sequence
        3. Classification head
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        cnn_channels: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(num_classes=num_classes)

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, cnn_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        # Project CNN features to transformer dimension
        self.feature_proj = nn.Linear(cnn_channels, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 32, d_model))  # Approx sequence length after CNN

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.cnn(x)  # [batch, cnn_channels, seq_len]

        # Transpose for transformer: [batch, seq_len, cnn_channels]
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.feature_proj(x)  # [batch, seq_len, d_model]

        # Add positional embedding (trim/pad if needed)
        seq_len = x.size(1)
        pos_embed = self.pos_embed[:, :seq_len, :] if seq_len <= self.pos_embed.size(1) else \
            F.pad(self.pos_embed, (0, 0, 0, seq_len - self.pos_embed.size(1)))
        x = x + pos_embed

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


class AttentionAugmentedCNN(BaseModel):
    """
    CNN with attention mechanisms

    Uses attention gates to focus on important regions
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__(num_classes=num_classes)

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=16, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # Attention
        att_weights = self.attention(x)
        x = x * att_weights  # Apply attention

        # Pool and classify
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
