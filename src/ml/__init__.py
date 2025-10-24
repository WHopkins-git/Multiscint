"""
Machine Learning models for scintillator classification

Includes traditional ML, deep learning, physics-informed models,
transformers, wavelet networks, and hybrid architectures.
"""

from .models import BaseModel
from .traditional_ml import TraditionalMLClassifier
from .cnn_models import SimpleCNN, ResNet1D
from .physics_informed import PhysicsInformedCNN, PhysicsLoss
from .transformer_models import WaveformTransformer, VisionTransformerWaveform
from .wavelet_models import WaveletScatteringClassifier
from .hybrid_models import CNNTransformerHybrid
from .training import ModelTrainer
from .evaluation import ModelEvaluator, ModelComparison
from .interpretability import ModelInterpretability

__all__ = [
    'BaseModel',
    'TraditionalMLClassifier',
    'SimpleCNN',
    'ResNet1D',
    'PhysicsInformedCNN',
    'PhysicsLoss',
    'WaveformTransformer',
    'VisionTransformerWaveform',
    'WaveletScatteringClassifier',
    'CNNTransformerHybrid',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelComparison',
    'ModelInterpretability'
]
