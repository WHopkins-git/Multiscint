"""Base model class"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Base class for all models"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def reset_parameters(self):
        """Reset all parameters"""
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
