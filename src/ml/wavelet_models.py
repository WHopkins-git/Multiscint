"""
Wavelet Scattering Networks for waveform classification

Uses multi-scale wavelet transform for interpretable feature extraction
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional


class WaveletScatteringClassifier:
    """
    Wavelet scattering transform + classical ML classifier

    More interpretable than deep learning while maintaining good performance
    Uses PyWavelets for wavelet transform
    """

    def __init__(
        self,
        J: int = 6,
        Q: int = 8,
        classifier_type: str = 'svm',
        **classifier_kwargs
    ):
        """
        Parameters:
            J: Number of scales (2^J = maximum scale)
            Q: Number of wavelets per octave
            classifier_type: 'svm' or 'random_forest'
            **classifier_kwargs: Arguments for the classifier
        """
        self.J = J
        self.Q = Q
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(
                C=classifier_kwargs.get('C', 10),
                kernel=classifier_kwargs.get('kernel', 'rbf'),
                gamma=classifier_kwargs.get('gamma', 'scale'),
                probability=True
            )
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=classifier_kwargs.get('n_estimators', 100),
                max_depth=classifier_kwargs.get('max_depth', 20),
                random_state=classifier_kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def wavelet_transform(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Apply wavelet scattering transform

        Parameters:
            waveforms: Array of waveforms, shape (N, samples)

        Returns:
            Scattering coefficients, shape (N, num_coeffs)
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets not installed. Install with: pip install PyWavelets")

        N = waveforms.shape[0]
        features = []

        for waveform in waveforms:
            # Multi-scale wavelet decomposition
            coeffs_list = []

            for j in range(1, self.J + 1):
                level = min(j, pywt.dwt_max_level(len(waveform), 'db4'))
                if level > 0:
                    coeffs = pywt.wavedec(waveform, 'db4', level=level)

                    # Extract features from each level
                    for coeff in coeffs:
                        if len(coeff) > 0:
                            # Statistical features
                            coeffs_list.extend([
                                np.mean(coeff),
                                np.std(coeff),
                                np.max(np.abs(coeff)),
                                np.sum(coeff ** 2)  # Energy
                            ])

            features.append(coeffs_list)

        # Ensure all feature vectors have same length
        max_len = max(len(f) for f in features)
        features_padded = [f + [0] * (max_len - len(f)) for f in features]

        return np.array(features_padded)

    def fit(self, waveforms: np.ndarray, labels: np.ndarray):
        """
        Train the classifier

        Parameters:
            waveforms: Array of waveforms, shape (N, samples)
            labels: Class labels, shape (N,)
        """
        # Extract wavelet features
        features = self.wavelet_transform(waveforms)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train classifier
        self.classifier.fit(features_scaled, labels)
        self.is_fitted = True

    def predict(self, waveforms: np.ndarray) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        features = self.wavelet_transform(waveforms)
        features_scaled = self.scaler.transform(features)

        return self.classifier.predict(features_scaled)

    def predict_proba(self, waveforms: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        features = self.wavelet_transform(waveforms)
        features_scaled = self.scaler.transform(features)

        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(features_scaled)
        else:
            raise ValueError("Classifier does not support probability prediction")

    def score(self, waveforms: np.ndarray, labels: np.ndarray) -> float:
        """Calculate accuracy"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        predictions = self.predict(waveforms)
        return np.mean(predictions == labels)


class SimpleWaveletNet(nn.Module):
    """
    Simple neural network with wavelet-inspired architecture

    Uses dilated convolutions at multiple scales
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        base_channels: int = 32
    ):
        super().__init__()

        self.num_classes = num_classes

        # Multi-scale dilated convolutions (wavelet-like)
        self.conv_scale1 = nn.Conv1d(1, base_channels, kernel_size=16, dilation=1, padding=7)
        self.conv_scale2 = nn.Conv1d(1, base_channels, kernel_size=16, dilation=2, padding=15)
        self.conv_scale4 = nn.Conv1d(1, base_channels, kernel_size=16, dilation=4, padding=30)
        self.conv_scale8 = nn.Conv1d(1, base_channels, kernel_size=16, dilation=8, padding=60)

        # Combine scales
        self.combine = nn.Conv1d(base_channels * 4, 64, kernel_size=1)

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Multi-scale features
        s1 = torch.relu(self.conv_scale1(x))
        s2 = torch.relu(self.conv_scale2(x))
        s4 = torch.relu(self.conv_scale4(x))
        s8 = torch.relu(self.conv_scale8(x))

        # Concatenate (need to match dimensions)
        min_len = min(s1.size(2), s2.size(2), s4.size(2), s8.size(2))
        s1 = s1[:, :, :min_len]
        s2 = s2[:, :, :min_len]
        s4 = s4[:, :, :min_len]
        s8 = s8[:, :, :min_len]

        combined = torch.cat([s1, s2, s4, s8], dim=1)

        # Combine scales
        x = torch.relu(self.combine(combined))

        # Pool and classify
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
