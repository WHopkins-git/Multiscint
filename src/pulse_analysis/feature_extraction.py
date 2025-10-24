"""
Pulse shape feature extraction for machine learning
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from typing import Dict, Optional
import warnings


class PulseFeatureExtractor:
    """
    Extract comprehensive features from pulse waveforms

    Parameters:
        sampling_rate_MHz: Sampling rate in MHz (default: 125)
        baseline_samples: Number of samples to use for baseline (default: 50)
    """

    def __init__(self, sampling_rate_MHz: float = 125.0, baseline_samples: int = 50):
        self.sampling_rate_MHz = sampling_rate_MHz
        self.dt_ns = 1000.0 / sampling_rate_MHz
        self.baseline_samples = baseline_samples

    def extract_features(self, waveform: np.ndarray, baseline: Optional[float] = None) -> Dict[str, float]:
        """
        Extract all features from a waveform

        Parameters:
            waveform: ADC waveform array
            baseline: Baseline value (computed if None)

        Returns:
            Dictionary of features
        """
        # Calculate baseline if not provided
        if baseline is None:
            baseline = np.mean(waveform[:self.baseline_samples])

        # Baseline-corrected waveform
        waveform_corrected = waveform - baseline

        features = {}

        # Amplitude features
        features['amplitude'] = np.max(waveform_corrected)
        features['baseline'] = baseline
        features['baseline_std'] = np.std(waveform[:self.baseline_samples])

        # Peak position
        peak_idx = np.argmax(waveform_corrected)
        features['peak_position'] = peak_idx * self.dt_ns

        # Timing features
        features['rise_time_10_90'] = self._calculate_rise_time(waveform_corrected, peak_idx)
        features['fall_time_90_10'] = self._calculate_fall_time(waveform_corrected, peak_idx)

        # Charge integration
        features['total_charge'] = np.sum(waveform_corrected[waveform_corrected > 0])

        # Tail charge (for PSD)
        tail_start = peak_idx + int(50 / self.dt_ns)  # 50 ns after peak
        if tail_start < len(waveform_corrected):
            features['tail_charge'] = np.sum(waveform_corrected[tail_start:])
            features['tail_total_ratio'] = features['tail_charge'] / features['total_charge'] if features['total_charge'] > 0 else 0
        else:
            features['tail_charge'] = 0
            features['tail_total_ratio'] = 0

        # Width (FWHM)
        features['width_fwhm'] = self._calculate_fwhm(waveform_corrected, peak_idx)

        # Shape features
        pulse_region = waveform_corrected[max(0, peak_idx - 50):min(len(waveform_corrected), peak_idx + 200)]
        if len(pulse_region) > 10:
            features['skewness'] = float(stats.skew(pulse_region))
            features['kurtosis'] = float(stats.kurtosis(pulse_region))
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0

        # Decay constant (simple exponential fit to tail)
        features['decay_constant'] = self._estimate_decay_constant(waveform_corrected, peak_idx)

        # Rise/fall slopes
        if features['rise_time_10_90'] > 0:
            features['rise_slope'] = (0.8 * features['amplitude']) / features['rise_time_10_90']
        else:
            features['rise_slope'] = 0

        if features['fall_time_90_10'] > 0:
            features['fall_slope'] = (0.8 * features['amplitude']) / features['fall_time_90_10']
        else:
            features['fall_slope'] = 0

        # Frequency domain features
        dominant_freq = self._calculate_dominant_frequency(waveform_corrected)
        features['dominant_frequency'] = dominant_freq

        return features

    def _calculate_rise_time(self, waveform: np.ndarray, peak_idx: int) -> float:
        """Calculate 10-90% rise time in nanoseconds"""
        amplitude = waveform[peak_idx]

        if amplitude <= 0:
            return 0

        thresh_10 = 0.1 * amplitude
        thresh_90 = 0.9 * amplitude

        # Find crossings before peak
        rising_edge = waveform[:peak_idx]

        # Find 10% crossing
        idx_10 = np.where(rising_edge >= thresh_10)[0]
        if len(idx_10) == 0:
            return 0
        t_10 = idx_10[0]

        # Find 90% crossing
        idx_90 = np.where(rising_edge >= thresh_90)[0]
        if len(idx_90) == 0:
            return 0
        t_90 = idx_90[0]

        rise_time = (t_90 - t_10) * self.dt_ns
        return max(0, rise_time)

    def _calculate_fall_time(self, waveform: np.ndarray, peak_idx: int) -> float:
        """Calculate 90-10% fall time in nanoseconds"""
        amplitude = waveform[peak_idx]

        if amplitude <= 0 or peak_idx >= len(waveform) - 1:
            return 0

        thresh_90 = 0.9 * amplitude
        thresh_10 = 0.1 * amplitude

        # Falling edge after peak
        falling_edge = waveform[peak_idx:]

        # Find 90% crossing
        idx_90 = np.where(falling_edge <= thresh_90)[0]
        if len(idx_90) == 0:
            return 0
        t_90 = idx_90[0]

        # Find 10% crossing
        idx_10 = np.where(falling_edge <= thresh_10)[0]
        if len(idx_10) == 0:
            return (len(falling_edge) - t_90) * self.dt_ns
        t_10 = idx_10[0]

        fall_time = (t_10 - t_90) * self.dt_ns
        return max(0, fall_time)

    def _calculate_fwhm(self, waveform: np.ndarray, peak_idx: int) -> float:
        """Calculate full width at half maximum in nanoseconds"""
        amplitude = waveform[peak_idx]

        if amplitude <= 0:
            return 0

        half_max = 0.5 * amplitude

        # Find crossings
        above_half = waveform >= half_max

        if not np.any(above_half):
            return 0

        # Find first and last crossing
        crossings = np.where(above_half)[0]
        if len(crossings) < 2:
            return 0

        fwhm = (crossings[-1] - crossings[0]) * self.dt_ns
        return fwhm

    def _estimate_decay_constant(self, waveform: np.ndarray, peak_idx: int) -> float:
        """Estimate exponential decay constant in nanoseconds"""
        if peak_idx >= len(waveform) - 10:
            return 0

        amplitude = waveform[peak_idx]

        if amplitude <= 0:
            return 0

        # Fit exponential to tail
        tail_start = peak_idx + int(10 / self.dt_ns)  # Start 10 ns after peak
        tail_end = min(len(waveform), peak_idx + int(500 / self.dt_ns))

        if tail_end <= tail_start:
            return 0

        tail = waveform[tail_start:tail_end]
        tail = np.maximum(tail, 1e-6)  # Avoid log(0)

        # Log-linear fit
        t = np.arange(len(tail)) * self.dt_ns

        # Only fit if we have positive values
        positive_mask = tail > 0
        if np.sum(positive_mask) < 5:
            return 0

        try:
            # Linear fit to log(tail)
            coeffs = np.polyfit(t[positive_mask], np.log(tail[positive_mask]), 1)
            decay_constant = -1.0 / coeffs[0] if coeffs[0] < 0 else 0
            return max(0, decay_constant)
        except:
            return 0

    def _calculate_dominant_frequency(self, waveform: np.ndarray) -> float:
        """Calculate dominant frequency in MHz"""
        # FFT
        n = len(waveform)
        fft_vals = fft(waveform)
        freqs = fftfreq(n, 1 / (self.sampling_rate_MHz * 1e6))  # Convert to Hz

        # Power spectrum (positive frequencies only)
        power = np.abs(fft_vals[:n // 2]) ** 2
        freqs_positive = freqs[:n // 2]

        # Find dominant frequency (excluding DC)
        if len(power) > 1:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_freq = freqs_positive[dominant_idx] / 1e6  # Convert to MHz
            return dominant_freq
        else:
            return 0


def extract_features_batch(waveforms: np.ndarray, extractor: PulseFeatureExtractor) -> np.ndarray:
    """
    Extract features from batch of waveforms

    Parameters:
        waveforms: Array of shape (N, M) where N=number of waveforms, M=samples
        extractor: PulseFeatureExtractor instance

    Returns:
        Feature array of shape (N, num_features)
    """
    feature_list = []

    for waveform in waveforms:
        features = extractor.extract_features(waveform)
        feature_list.append(list(features.values()))

    return np.array(feature_list)
