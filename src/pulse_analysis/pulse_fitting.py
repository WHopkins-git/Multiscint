"""Pulse fitting functions"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict
import warnings


def exponential_decay(t: np.ndarray, A: float, tau: float, baseline: float = 0) -> np.ndarray:
    """Single exponential decay: A * exp(-t/tau) + baseline"""
    return A * np.exp(-t / tau) + baseline


def double_exponential_decay(
    t: np.ndarray,
    A1: float,
    tau1: float,
    A2: float,
    tau2: float,
    baseline: float = 0
) -> np.ndarray:
    """Double exponential decay (for LYSO)"""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline


class PulseFitter:
    """Fit pulse shapes to extract decay constants"""

    def __init__(self, sampling_rate_MHz: float = 125.0):
        self.dt_ns = 1000.0 / sampling_rate_MHz

    def fit_single_exponential(self, waveform: np.ndarray, peak_idx: int) -> Dict:
        """Fit single exponential to pulse tail"""
        # Extract tail region
        tail_start = peak_idx + 5
        tail = waveform[tail_start:]

        if len(tail) < 10:
            return {'success': False, 'error': 'Insufficient tail length'}

        t = np.arange(len(tail)) * self.dt_ns
        amplitude = waveform[peak_idx]

        try:
            p0 = [amplitude, 50, 0]  # Initial: A, tau=50ns, baseline=0
            bounds = ([0, 1, -amplitude * 0.1], [amplitude * 2, 1000, amplitude * 0.1])

            popt, pcov = curve_fit(exponential_decay, t, tail, p0=p0, bounds=bounds, maxfev=5000)

            return {
                'success': True,
                'amplitude': popt[0],
                'tau': popt[1],
                'baseline': popt[2],
                'errors': np.sqrt(np.diag(pcov))
            }
        except:
            return {'success': False, 'error': 'Fit failed'}

    def fit_double_exponential(self, waveform: np.ndarray, peak_idx: int) -> Dict:
        """Fit double exponential (for LYSO)"""
        tail_start = peak_idx + 5
        tail = waveform[tail_start:]

        if len(tail) < 20:
            return {'success': False, 'error': 'Insufficient tail length'}

        t = np.arange(len(tail)) * self.dt_ns
        amplitude = waveform[peak_idx]

        try:
            # Initial guess: fast (40ns) + slow (200ns) components
            p0 = [amplitude * 0.7, 40, amplitude * 0.3, 200, 0]
            bounds = (
                [0, 10, 0, 100, -amplitude * 0.1],
                [amplitude, 100, amplitude, 500, amplitude * 0.1]
            )

            popt, pcov = curve_fit(double_exponential_decay, t, tail, p0=p0, bounds=bounds, maxfev=10000)

            return {
                'success': True,
                'A1': popt[0],
                'tau1': popt[1],
                'A2': popt[2],
                'tau2': popt[3],
                'baseline': popt[4],
                'errors': np.sqrt(np.diag(pcov))
            }
        except:
            return {'success': False, 'error': 'Fit failed'}
