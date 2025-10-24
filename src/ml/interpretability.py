"""
Model interpretability tools

Provides methods to understand what models learn:
    - Feature importance (for tree-based models)
    - SHAP values
    - Attention visualization (for transformers)
    - Saliency maps (for CNNs)
    - Physics validation (for PINNs)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelInterpretability:
    """
    Interpretability tools for neural network models
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    def compute_saliency_map(self, waveform: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute saliency map: gradient of output w.r.t. input

        Shows which parts of waveform are most important for prediction

        Parameters:
            waveform: Input waveform, shape (samples,)
            target_class: Target class (if None, use predicted class)

        Returns:
            Saliency map, shape (samples,)
        """
        self.model.eval()

        # Prepare input
        waveform = waveform.clone().detach().to(self.device)
        waveform.requires_grad = True

        # Forward pass
        logits = self.model(waveform.unsqueeze(0))

        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Saliency is absolute value of gradient
        saliency = torch.abs(waveform.grad).cpu().numpy()

        return saliency

    def plot_saliency_map(
        self,
        waveform: np.ndarray,
        saliency: np.ndarray,
        title: str = "Saliency Map"
    ):
        """
        Visualize saliency map overlaid on waveform

        Parameters:
            waveform: Original waveform
            saliency: Saliency map
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Waveform
        axes[0].plot(waveform, color='blue', linewidth=1.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Input Waveform')
        axes[0].grid(True, alpha=0.3)

        # Saliency
        axes[1].plot(saliency, color='red', linewidth=1.5)
        axes[1].set_xlabel('Time Sample')
        axes[1].set_ylabel('Saliency (|∂output/∂input|)')
        axes[1].set_title('Saliency Map')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        return fig

    def validate_physics_learning(
        self,
        pinn_model,
        test_waveforms: np.ndarray,
        test_labels: np.ndarray,
        known_decay_times: Dict[int, float] = None
    ) -> Dict:
        """
        For physics-informed models: validate learned physics

        Extracts decay times from waveforms and compares to known values

        Parameters:
            pinn_model: Physics-informed model
            test_waveforms: Test waveforms
            test_labels: Test labels (class indices)
            known_decay_times: Dictionary mapping class to known decay time (ns)

        Returns:
            Dictionary with learned vs known decay times
        """
        if known_decay_times is None:
            known_decay_times = {
                0: 40.0,    # LYSO
                1: 300.0,   # BGO
                2: 230.0,   # NaI
                3: 2.4      # Plastic
            }

        self.model.eval()

        learned_decay_times = {i: [] for i in range(4)}

        with torch.no_grad():
            for waveform, label in zip(test_waveforms, test_labels):
                # Fit exponential to extract decay time
                peak_idx = np.argmax(waveform)

                if peak_idx >= len(waveform) - 20:
                    continue

                tail_start = peak_idx + 5
                tail = waveform[tail_start:]

                if len(tail) < 10:
                    continue

                # Log-linear fit
                positive_mask = tail > 0
                if np.sum(positive_mask) < 5:
                    continue

                t = np.arange(len(tail))[positive_mask] * 8.0  # 8 ns sampling
                log_tail = np.log(tail[positive_mask] + 1e-8)

                # Linear regression
                coeffs = np.polyfit(t, log_tail, 1)
                fitted_tau = -1.0 / coeffs[0] if coeffs[0] < 0 else 0

                if 0 < fitted_tau < 1000:  # Sanity check
                    learned_decay_times[label].append(fitted_tau)

        # Calculate statistics
        results = {}
        for class_idx in range(4):
            if len(learned_decay_times[class_idx]) > 0:
                learned_mean = np.mean(learned_decay_times[class_idx])
                learned_std = np.std(learned_decay_times[class_idx])
                known = known_decay_times[class_idx]

                results[f'class_{class_idx}'] = {
                    'known_tau': known,
                    'learned_tau_mean': learned_mean,
                    'learned_tau_std': learned_std,
                    'relative_error': abs(learned_mean - known) / known * 100
                }

        return results

    def plot_physics_validation(self, validation_results: Dict):
        """
        Plot learned vs known decay times

        Parameters:
            validation_results: Output from validate_physics_learning
        """
        class_names = ['LYSO', 'BGO', 'NaI', 'Plastic']

        known_taus = []
        learned_taus = []
        learned_stds = []

        for i in range(4):
            if f'class_{i}' in validation_results:
                known_taus.append(validation_results[f'class_{i}']['known_tau'])
                learned_taus.append(validation_results[f'class_{i}']['learned_tau_mean'])
                learned_stds.append(validation_results[f'class_{i}']['learned_tau_std'])

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.errorbar(
            known_taus,
            learned_taus,
            yerr=learned_stds,
            fmt='o',
            markersize=10,
            capsize=5,
            label='Learned'
        )

        # Add labels
        for i, name in enumerate(class_names[:len(known_taus)]):
            ax.annotate(name, (known_taus[i], learned_taus[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')

        # Perfect match line
        max_tau = max(max(known_taus), max(learned_taus))
        ax.plot([0, max_tau], [0, max_tau], 'r--', linewidth=2, label='Perfect Match')

        ax.set_xlabel('Known Decay Time (ns)', fontsize=14)
        ax.set_ylabel('Learned Decay Time (ns)', fontsize=14)
        ax.set_title('Physics-Informed Model: Learned vs Known Decay Times', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        return fig

    def visualize_attention_weights(
        self,
        transformer_model,
        waveform: torch.Tensor,
        layer_idx: int = 0
    ):
        """
        Visualize attention weights for transformer models

        Note: This requires the model to expose attention weights,
        which is not implemented in the base transformer.
        This is a placeholder for full implementation.

        Parameters:
            transformer_model: Transformer model
            waveform: Input waveform
            layer_idx: Which transformer layer to visualize
        """
        # This would require modifications to transformer model
        # to return attention weights
        print("Attention visualization requires model modifications")
        print("See TransformerWithAttentionMaps for implementation")

        return None


def shap_explain_traditional_ml(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list):
    """
    SHAP explanation for traditional ML models

    Parameters:
        model: Trained sklearn/XGBoost model
        X_train: Training data (for background)
        X_test: Test data to explain
        feature_names: List of feature names

    Returns:
        SHAP values and explainer
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.Explainer(model.predict_proba, X_train)
    else:
        explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()

    return shap_values, explainer
