"""
Energy calibration module for gamma spectroscopy
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import warnings


class EnergyCalibrator:
    """
    Perform energy calibration for scintillation detectors

    Fits a linear relationship between ADC channel and energy:
        Energy (keV) = slope * Channel + intercept

    Parameters:
        None

    Attributes:
        slope: Calibration slope (keV/channel)
        intercept: Calibration intercept (keV)
        r_squared: Goodness of fit
        residuals: Calibration residuals
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_squared = None
        self.residuals = None
        self.known_energies = None
        self.measured_channels = None

    def calibrate(
        self,
        measured_peaks: np.ndarray,
        known_energies: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Perform energy calibration

        Parameters:
            measured_peaks: Measured peak positions (ADC channels)
            known_energies: Known gamma energies (keV)
            weights: Optional weights for each peak

        Returns:
            (slope, intercept, r_squared)
        """
        measured_peaks = np.asarray(measured_peaks)
        known_energies = np.asarray(known_energies)

        if len(measured_peaks) != len(known_energies):
            raise ValueError("Number of measured peaks must match known energies")

        if len(measured_peaks) < 2:
            raise ValueError("Need at least 2 points for calibration")

        # Linear regression
        if weights is not None:
            # Weighted least squares
            W = np.diag(weights)
            X = np.column_stack([measured_peaks, np.ones(len(measured_peaks))])
            params = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ known_energies
            self.slope, self.intercept = params
        else:
            # Ordinary least squares
            self.slope, self.intercept, r_value, p_value, std_err = stats.linregress(
                measured_peaks, known_energies
            )
            self.r_squared = r_value ** 2

        # Calculate residuals
        predicted = self.slope * measured_peaks + self.intercept
        self.residuals = known_energies - predicted

        # Calculate R² if not already done
        if self.r_squared is None:
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((known_energies - np.mean(known_energies)) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot)

        self.known_energies = known_energies
        self.measured_channels = measured_peaks

        return self.slope, self.intercept, self.r_squared

    def apply_calibration(
        self,
        channels: np.ndarray
    ) -> np.ndarray:
        """
        Apply calibration to convert channels to energy

        Parameters:
            channels: ADC channel values

        Returns:
            Calibrated energies in keV
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Must run calibrate() before applying calibration")

        return self.slope * channels + self.intercept

    def inverse_calibration(
        self,
        energies: np.ndarray
    ) -> np.ndarray:
        """
        Convert energies back to channels

        Parameters:
            energies: Energy values in keV

        Returns:
            ADC channel values
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Must run calibrate() before inverse calibration")

        return (energies - self.intercept) / self.slope

    def get_calibration_summary(self) -> Dict:
        """Get calibration summary as dictionary"""
        if self.slope is None:
            return {"calibrated": False}

        return {
            "calibrated": True,
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "r_squared": float(self.r_squared),
            "n_points": len(self.measured_channels),
            "residuals_mean": float(np.mean(self.residuals)),
            "residuals_std": float(np.std(self.residuals)),
            "residuals_max": float(np.max(np.abs(self.residuals)))
        }

    def calculate_resolution(
        self,
        peak_channel: float,
        fwhm_channels: float
    ) -> Tuple[float, float]:
        """
        Calculate energy resolution

        Parameters:
            peak_channel: Peak centroid (ADC channels)
            fwhm_channels: FWHM of peak (ADC channels)

        Returns:
            (resolution_keV, resolution_percent)
        """
        if self.slope is None:
            raise ValueError("Must run calibrate() first")

        # Convert to energy
        peak_energy = self.apply_calibration(np.array([peak_channel]))[0]
        fwhm_energy = fwhm_channels * self.slope

        # Resolution as percentage
        resolution_percent = (fwhm_energy / peak_energy) * 100

        return fwhm_energy, resolution_percent

    def save_calibration(self, file_path: str) -> None:
        """Save calibration parameters to JSON file"""
        import json

        summary = self.get_calibration_summary()

        # Add calibration points
        if self.measured_channels is not None:
            summary['calibration_points'] = {
                'measured_channels': self.measured_channels.tolist(),
                'known_energies': self.known_energies.tolist(),
                'residuals': self.residuals.tolist()
            }

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_calibration(self, file_path: str) -> None:
        """Load calibration parameters from JSON file"""
        import json

        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data.get('calibrated', False):
            raise ValueError("File does not contain calibration data")

        self.slope = data['slope']
        self.intercept = data['intercept']
        self.r_squared = data['r_squared']

        if 'calibration_points' in data:
            self.measured_channels = np.array(data['calibration_points']['measured_channels'])
            self.known_energies = np.array(data['calibration_points']['known_energies'])
            self.residuals = np.array(data['calibration_points']['residuals'])


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float, background: float = 0) -> np.ndarray:
    """Gaussian function for peak fitting"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + background


def gaussian_with_tail(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    tail_amplitude: float,
    tail_slope: float,
    background: float = 0
) -> np.ndarray:
    """
    Gaussian with low-energy tail for scintillator peaks

    Models incomplete charge collection
    """
    gaussian_part = amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    # Exponential tail on low-energy side
    tail_part = np.zeros_like(x)
    low_energy_mask = x < mean
    tail_part[low_energy_mask] = tail_amplitude * np.exp(tail_slope * (x[low_energy_mask] - mean))

    return gaussian_part + tail_part + background
