"""Pile-up detection algorithms"""

import numpy as np
from scipy.optimize import curve_fit


class PileupDetector:
    """Detect pile-up events in waveforms"""

    def __init__(self, sampling_rate_MHz: float = 125.0):
        self.dt_ns = 1000.0 / sampling_rate_MHz

    def detect_baseline_restoration(
        self,
        waveform: np.ndarray,
        baseline: float,
        threshold_samples: int = 50
    ) -> bool:
        """Detect pile-up by checking baseline restoration"""
        tail = waveform[-threshold_samples:]
        baseline_restored = np.abs(np.mean(tail) - baseline) < 3 * np.std(tail)
        return not baseline_restored

    def detect_fit_quality(self, waveform: np.ndarray, decay_time_ns: float, peak_idx: int) -> bool:
        """Detect pile-up by checking exponential fit quality"""
        if peak_idx >= len(waveform) - 10:
            return False

        tail = waveform[peak_idx:]
        t = np.arange(len(tail)) * self.dt_ns
        amplitude = waveform[peak_idx]

        try:
            def exp_model(t, A, tau, bg):
                return A * np.exp(-t / tau) + bg

            p0 = [amplitude, decay_time_ns, 0]
            popt, _ = curve_fit(exp_model, t, tail, p0=p0, maxfev=1000)

            fitted = exp_model(t, *popt)
            residuals = tail - fitted
            chi_square = np.sum(residuals ** 2) / len(tail)

            # High chi-square indicates pile-up
            return chi_square > amplitude * 0.1
        except:
            return False

    def detect_derivative_anomaly(self, waveform: np.ndarray) -> bool:
        """Detect pile-up using second derivative"""
        d2 = np.diff(waveform, n=2)
        anomaly_score = np.max(np.abs(d2))
        threshold = 3 * np.std(d2)
        return anomaly_score > threshold
