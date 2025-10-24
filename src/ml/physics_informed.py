"""
Physics-Informed Neural Networks (PINNs) for scintillator classification

Incorporates physical constraints:
    - Exponential decay times (τ = 2.4, 40, 230, 300 ns for different scintillators)
    - Energy conservation (integral = amplitude)
    - Rise time consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_models import SimpleCNN
from typing import Dict, Tuple


class PhysicsLoss(nn.Module):
    """
    Custom loss function incorporating physics constraints

    Parameters:
        decay_times: Dictionary mapping class index to decay time (ns)
        alpha: Weight for classification loss
        beta: Weight for decay time loss
        gamma: Weight for energy conservation loss
    """

    def __init__(
        self,
        decay_times: Dict[int, float] = None,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        dt_ns: float = 8.0
    ):
        super().__init__()

        # Default decay times for each scintillator class
        if decay_times is None:
            decay_times = {
                0: 40.0,    # LYSO
                1: 300.0,   # BGO
                2: 230.0,   # NaI
                3: 2.4      # Plastic
            }

        self.decay_times = decay_times
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt_ns = dt_ns

    def decay_time_loss(
        self,
        waveforms: torch.Tensor,
        predicted_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize if waveform decay doesn't match expected tau

        Parameters:
            waveforms: Input waveforms, shape (batch, samples)
            predicted_classes: Predicted class indices, shape (batch,)

        Returns:
            Decay time loss (scalar)
        """
        batch_size = waveforms.shape[0]
        device = waveforms.device

        # Find peak positions
        peak_indices = torch.argmax(waveforms, dim=1)

        losses = []

        for i in range(batch_size):
            waveform = waveforms[i]
            peak_idx = peak_indices[i].item()
            predicted_class = predicted_classes[i].item()

            # Skip if peak too close to end
            if peak_idx >= len(waveform) - 20:
                continue

            # Extract tail (20 samples after peak)
            tail_start = peak_idx + 5
            tail_end = min(peak_idx + 200, len(waveform))
            tail = waveform[tail_start:tail_end]

            if len(tail) < 10:
                continue

            # Create time array
            t = torch.arange(len(tail), device=device, dtype=torch.float32) * self.dt_ns

            # Expected decay constant for this class
            expected_tau = self.decay_times.get(predicted_class, 50.0)

            # Fit exponential: log(tail) = log(A) - t/tau
            # Only fit positive values
            positive_mask = tail > 0

            if torch.sum(positive_mask) < 5:
                continue

            log_tail = torch.log(tail[positive_mask] + 1e-8)
            t_positive = t[positive_mask]

            # Linear regression to get tau
            # y = a + b*t  where y = log(tail), b = -1/tau
            t_mean = t_positive.mean()
            y_mean = log_tail.mean()

            numerator = torch.sum((t_positive - t_mean) * (log_tail - y_mean))
            denominator = torch.sum((t_positive - t_mean) ** 2)

            if denominator > 0:
                slope = numerator / denominator
                fitted_tau = -1.0 / (slope + 1e-8)

                # Loss: squared difference between fitted and expected tau
                tau_loss = (fitted_tau - expected_tau) ** 2
                losses.append(tau_loss)

        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device)

    def energy_conservation_loss(
        self,
        waveforms: torch.Tensor,
        amplitudes: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure total charge (integral) is proportional to amplitude

        Parameters:
            waveforms: Baseline-corrected waveforms, shape (batch, samples)
            amplitudes: Peak amplitudes, shape (batch,)

        Returns:
            Energy conservation loss (scalar)
        """
        # Calculate integrated charge
        integrated_charge = torch.sum(torch.clamp(waveforms, min=0), dim=1)

        # Loss: integrated charge should be proportional to amplitude
        # Normalize by amplitude
        ratio = integrated_charge / (amplitudes + 1e-8)

        # We want consistent ratios, so minimize variance
        loss = torch.var(ratio)

        return loss

    def rise_time_consistency_loss(
        self,
        waveforms: torch.Tensor,
        predicted_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure rise times are consistent with scintillator type

        Fast scintillators (Plastic, LYSO) should have fast rise times
        Slow scintillators (BGO, NaI) can have slower rise times
        """
        batch_size = waveforms.shape[0]
        device = waveforms.device

        expected_rise_times = {
            0: 5.0,    # LYSO - fast
            1: 20.0,   # BGO - slow
            2: 10.0,   # NaI - medium
            3: 2.0     # Plastic - very fast
        }

        losses = []

        for i in range(batch_size):
            waveform = waveforms[i]
            predicted_class = predicted_classes[i].item()

            # Find peak
            peak_idx = torch.argmax(waveform).item()
            amplitude = waveform[peak_idx]

            if peak_idx < 10:
                continue

            # Calculate 10-90% rise time
            thresh_10 = 0.1 * amplitude
            thresh_90 = 0.9 * amplitude

            rising_edge = waveform[:peak_idx]

            # Find crossings
            above_10 = rising_edge >= thresh_10
            above_90 = rising_edge >= thresh_90

            if torch.sum(above_10) > 0 and torch.sum(above_90) > 0:
                idx_10 = torch.where(above_10)[0][0].float()
                idx_90 = torch.where(above_90)[0][0].float()

                rise_time = (idx_90 - idx_10) * self.dt_ns

                expected_rise = expected_rise_times.get(predicted_class, 10.0)

                # Soft constraint: penalize if rise time deviates significantly
                rt_loss = torch.abs(rise_time - expected_rise) / expected_rise
                losses.append(rt_loss)

        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        waveforms: torch.Tensor,
        amplitudes: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined physics-informed loss

        Parameters:
            logits: Model predictions, shape (batch, num_classes)
            labels: True labels, shape (batch,)
            waveforms: Input waveforms, shape (batch, samples)
            amplitudes: Peak amplitudes, shape (batch,)

        Returns:
            (total_loss, loss_components_dict)
        """
        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)

        # Get predicted classes
        predicted_classes = torch.argmax(logits, dim=1)

        # Physics losses
        decay_loss = self.decay_time_loss(waveforms, predicted_classes)
        energy_loss = self.energy_conservation_loss(waveforms, amplitudes)
        rise_loss = self.rise_time_consistency_loss(waveforms, predicted_classes)

        # Combined loss
        total_loss = (
            self.alpha * ce_loss +
            self.beta * decay_loss +
            self.gamma * energy_loss +
            0.05 * rise_loss  # Small weight for rise time
        )

        # Return loss components for monitoring
        loss_components = {
            'total': total_loss.item(),
            'classification': ce_loss.item(),
            'decay_time': decay_loss.item(),
            'energy_conservation': energy_loss.item(),
            'rise_time': rise_loss.item()
        }

        return total_loss, loss_components


class PhysicsInformedCNN(nn.Module):
    """
    CNN with physics-informed loss function

    Uses SimpleCNN architecture but trains with physics constraints
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 4,
        dropout: float = 0.3,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1
    ):
        super().__init__()

        # Base CNN model
        self.cnn = SimpleCNN(
            input_length=input_length,
            num_classes=num_classes,
            dropout=dropout
        )

        # Physics-informed loss
        self.physics_loss = PhysicsLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass through CNN"""
        return self.cnn(x)

    def compute_loss(
        self,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        amplitudes: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss

        Parameters:
            waveforms: Input waveforms, shape (batch, samples)
            labels: True labels, shape (batch,)
            amplitudes: Peak amplitudes, shape (batch,)

        Returns:
            (loss, loss_components)
        """
        # Forward pass
        logits = self.forward(waveforms)

        # Compute physics-informed loss
        loss, components = self.physics_loss(logits, labels, waveforms, amplitudes)

        return loss, components

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Get model size in MB"""
        return self.cnn.get_model_size_mb()

    def reset_parameters(self):
        """Reset all parameters"""
        self.cnn.reset_parameters()
