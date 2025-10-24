"""
Convolutional Neural Networks for waveform classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import BaseModel


class SimpleCNN(BaseModel):
    """
    Baseline 1D CNN for waveform classification

    Architecture:
        - 3 convolutional layers with max pooling
        - 2 fully connected layers
        - Dropout for regularization
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__(num_classes=num_classes)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=16, stride=2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=1)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate flattened size
        self._calculate_flatten_size(input_length)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_classes)

    def _calculate_flatten_size(self, input_length):
        """Calculate size after convolutions and pooling"""
        x = torch.zeros(1, 1, input_length)

        # Layer 1
        x = self.conv1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.pool3(x)

        self.flatten_size = x.numel()

    def forward(self, x):
        """
        Forward pass

        Parameters:
            x: Input tensor, shape (batch, samples) or (batch, 1, samples)

        Returns:
            Logits, shape (batch, num_classes)
        """
        # Ensure 3D input
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


class ResNet1D(BaseModel):
    """
    ResNet-style architecture for 1D waveforms

    Uses residual connections for better gradient flow
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        num_blocks: int = 3,
        base_channels: int = 32
    ):
        super().__init__(num_classes=num_classes)

        self.input_conv = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool = nn.MaxPool1d(2)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = base_channels

        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            self.res_blocks.append(
                ResidualBlock1D(in_channels, out_channels, stride=2 if i > 0 else 1)
            )
            in_channels = out_channels

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward pass"""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Initial conv
        x = F.relu(self.bn1(self.input_conv(x)))
        x = self.pool(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)

        return x


class ResidualBlock1D(nn.Module):
    """1D Residual block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.skip(residual)
        out = F.relu(out)

        return out
