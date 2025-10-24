"""Optical crosstalk analysis for SiPMs"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple


class CrosstalkAnalyzer:
    """Analyze optical crosstalk in SiPMs"""

    def __init__(self, n_cells: int = 18000):
        self.n_cells = n_cells

    def measure_crosstalk_vs_amplitude(
        self,
        amplitudes: np.ndarray,
        charges: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Measure crosstalk probability vs pulse amplitude

        Parameters:
            amplitudes: Peak amplitudes
            charges: Integrated charges
            n_bins: Number of amplitude bins

        Returns:
            Dictionary with bin centers and crosstalk probabilities
        """
        # Bin data by amplitude
        amp_bins = np.linspace(np.min(amplitudes), np.max(amplitudes), n_bins + 1)
        crosstalk_probs = []
        bin_centers = []

        for i in range(n_bins):
            mask = (amplitudes >= amp_bins[i]) & (amplitudes < amp_bins[i + 1])

            if np.sum(mask) < 10:
                continue

            amps = amplitudes[mask]
            chgs = charges[mask]

            # Expected charge without crosstalk
            expected_charge = np.mean(amps)

            # Measured charge
            measured_charge = np.mean(chgs)

            # Excess indicates crosstalk
            if expected_charge > 0:
                crosstalk_prob = (measured_charge / expected_charge - 1.0) * 100
                crosstalk_probs.append(max(0, crosstalk_prob))
                bin_centers.append((amp_bins[i] + amp_bins[i + 1]) / 2)

        return {
            'amplitude_bins': np.array(bin_centers),
            'crosstalk_probability': np.array(crosstalk_probs)
        }
