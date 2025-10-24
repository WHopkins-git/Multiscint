"""Pile-up correction algorithms"""

import numpy as np


class PileupCorrector:
    """Correct pile-up events"""

    def __init__(self, sampling_rate_MHz: float = 125.0):
        self.dt_ns = 1000.0 / sampling_rate_MHz

    def deconvolve_pulses(
        self,
        waveform: np.ndarray,
        decay_time_ns: float
    ) -> Tuple[bool, float, float, int]:
        """
        Attempt to separate overlapping pulses

        Returns:
            (success, energy1, energy2, delay_samples)
        """
        # Find first peak
        peak1_idx = np.argmax(waveform)
        peak1_amplitude = waveform[peak1_idx]

        # Model first pulse
        t = np.arange(len(waveform) - peak1_idx) * self.dt_ns
        model_pulse1 = peak1_amplitude * np.exp(-t / decay_time_ns)

        # Subtract
        residual = waveform.copy()
        residual[peak1_idx:] -= model_pulse1

        # Check for second pulse
        threshold = 5 * np.std(residual[:peak1_idx])

        if np.max(residual) > threshold:
            peak2_idx = np.argmax(residual)
            peak2_amplitude = residual[peak2_idx]

            energy1 = np.sum(model_pulse1)
            energy2 = np.sum(residual[residual > 0])

            return True, energy1, energy2, peak2_idx - peak1_idx
        else:
            return False, np.sum(waveform), 0, 0
