"""
Peak finding and fitting for gamma spectroscopy
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional
import warnings


class PeakFinder:
    """
    Find photopeaks in gamma spectra

    Parameters:
        min_prominence: Minimum peak prominence (relative to max)
        min_distance: Minimum distance between peaks (bins)
        min_height: Minimum peak height (relative to max)
    """

    def __init__(
        self,
        min_prominence: float = 0.05,
        min_distance: int = 20,
        min_height: float = 0.02
    ):
        self.min_prominence = min_prominence
        self.min_distance = min_distance
        self.min_height = min_height

    def find_peaks(
        self,
        spectrum: np.ndarray,
        bins: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Find peaks in spectrum

        Parameters:
            spectrum: Histogram counts
            bins: Bin edges (optional)

        Returns:
            List of dictionaries with peak information
        """
        if bins is None:
            bins = np.arange(len(spectrum))
        else:
            # Use bin centers
            bins = (bins[:-1] + bins[1:]) / 2

        # Normalize spectrum for peak finding
        spectrum_max = np.max(spectrum)

        # Find peaks
        peak_indices, properties = find_peaks(
            spectrum,
            prominence=self.min_prominence * spectrum_max,
            distance=self.min_distance,
            height=self.min_height * spectrum_max
        )

        # Calculate peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(
            spectrum,
            peak_indices,
            rel_height=0.5
        )

        # Collect peak information
        peaks = []
        for idx, peak_idx in enumerate(peak_indices):
            peak_info = {
                'index': int(peak_idx),
                'position': float(bins[peak_idx]),
                'height': float(spectrum[peak_idx]),
                'prominence': float(properties['prominences'][idx]),
                'width': float(widths[idx]),
                'fwhm': float(widths[idx] * np.mean(np.diff(bins))),
                'left_base': float(bins[int(left_ips[idx])]),
                'right_base': float(bins[int(right_ips[idx])])
            }
            peaks.append(peak_info)

        # Sort by height (descending)
        peaks.sort(key=lambda p: p['height'], reverse=True)

        return peaks

    def identify_peak(
        self,
        spectrum: np.ndarray,
        bins: np.ndarray,
        expected_position: float,
        search_range: float = 50
    ) -> Optional[Dict]:
        """
        Find specific peak near expected position

        Parameters:
            spectrum: Histogram counts
            bins: Bin centers
            expected_position: Expected peak position
            search_range: Search window (± range around expected position)

        Returns:
            Peak dictionary or None if not found
        """
        # Find all peaks
        all_peaks = self.find_peaks(spectrum, bins)

        # Find closest to expected position
        in_range = [
            p for p in all_peaks
            if abs(p['position'] - expected_position) < search_range
        ]

        if not in_range:
            return None

        # Return highest peak in range
        return max(in_range, key=lambda p: p['height'])


class GaussianFitter:
    """
    Fit Gaussian (+ background) to peaks

    Parameters:
        fit_range_fwhm: Fit range in units of FWHM (e.g., 3.0 = ±3 FWHM)
    """

    def __init__(self, fit_range_fwhm: float = 3.0):
        self.fit_range_fwhm = fit_range_fwhm

    def fit_peak(
        self,
        spectrum: np.ndarray,
        bins: np.ndarray,
        peak_info: Dict,
        background_order: int = 1
    ) -> Dict:
        """
        Fit Gaussian to peak

        Parameters:
            spectrum: Histogram counts
            bins: Bin centers
            peak_info: Peak information from PeakFinder
            background_order: 0=constant, 1=linear background

        Returns:
            Dictionary with fit results
        """
        # Extract fit region
        peak_pos = peak_info['position']
        fwhm = peak_info['fwhm']
        fit_range = self.fit_range_fwhm * fwhm

        mask = np.abs(bins - peak_pos) < fit_range
        x_data = bins[mask]
        y_data = spectrum[mask]

        if len(x_data) < 5:
            raise ValueError("Insufficient data points for fitting")

        # Initial parameters
        amplitude_init = peak_info['height']
        mean_init = peak_pos
        sigma_init = fwhm / 2.355  # Convert FWHM to sigma
        background_init = np.min(y_data)

        # Define fit function
        if background_order == 0:
            def fit_func(x, amp, mean, sigma, bg):
                return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + bg

            p0 = [amplitude_init, mean_init, sigma_init, background_init]
            bounds = (
                [0, peak_pos - fwhm, sigma_init * 0.5, 0],
                [amplitude_init * 2, peak_pos + fwhm, sigma_init * 2, amplitude_init * 0.5]
            )

        elif background_order == 1:
            def fit_func(x, amp, mean, sigma, bg0, bg1):
                return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + bg0 + bg1 * x

            p0 = [amplitude_init, mean_init, sigma_init, background_init, 0]
            bounds = (
                [0, peak_pos - fwhm, sigma_init * 0.5, 0, -0.1],
                [amplitude_init * 2, peak_pos + fwhm, sigma_init * 2, amplitude_init * 0.5, 0.1]
            )
        else:
            raise ValueError(f"background_order must be 0 or 1, got {background_order}")

        # Fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(fit_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)

            # Extract parameters
            amplitude, mean, sigma = popt[:3]
            perr = np.sqrt(np.diag(pcov))

            # Calculate fit quality
            y_fit = fit_func(x_data, *popt)
            residuals = y_data - y_fit
            chi_square = np.sum((residuals ** 2) / np.maximum(y_fit, 1))
            reduced_chi_square = chi_square / (len(x_data) - len(popt))

            # Calculate FWHM
            fwhm_fit = 2.355 * sigma
            fwhm_error = 2.355 * perr[2]

            # Calculate area (integral of Gaussian)
            area = amplitude * sigma * np.sqrt(2 * np.pi)
            area_error = area * np.sqrt((perr[0] / amplitude) ** 2 + (perr[2] / sigma) ** 2)

            return {
                'amplitude': float(amplitude),
                'amplitude_error': float(perr[0]),
                'mean': float(mean),
                'mean_error': float(perr[1]),
                'sigma': float(sigma),
                'sigma_error': float(perr[2]),
                'fwhm': float(fwhm_fit),
                'fwhm_error': float(fwhm_error),
                'area': float(area),
                'area_error': float(area_error),
                'chi_square': float(chi_square),
                'reduced_chi_square': float(reduced_chi_square),
                'fit_range': [float(x_data.min()), float(x_data.max())],
                'n_points': int(len(x_data)),
                'success': True
            }

        except Exception as e:
            warnings.warn(f"Fit failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def fit_multiple_peaks(
        self,
        spectrum: np.ndarray,
        bins: np.ndarray,
        peak_positions: List[float],
        background_order: int = 1
    ) -> List[Dict]:
        """
        Fit multiple peaks simultaneously

        Useful for overlapping peaks (e.g., Co-60 doublet)
        """
        # Define multi-Gaussian function
        def multi_gaussian(x, *params):
            n_peaks = len(peak_positions)
            result = np.zeros_like(x, dtype=float)

            # Each peak has 3 parameters: amplitude, mean, sigma
            for i in range(n_peaks):
                amp = params[i * 3]
                mean = params[i * 3 + 1]
                sigma = params[i * 3 + 2]
                result += amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

            # Add background
            if background_order == 0:
                result += params[-1]
            elif background_order == 1:
                result += params[-2] + params[-1] * x

            return result

        # Prepare initial parameters and bounds
        p0 = []
        lower_bounds = []
        upper_bounds = []

        for pos in peak_positions:
            # Find peak info
            idx = np.argmin(np.abs(bins - pos))
            amp = spectrum[idx]
            sigma = 10.0  # Initial guess

            p0.extend([amp, pos, sigma])
            lower_bounds.extend([0, pos - 50, 1])
            upper_bounds.extend([amp * 2, pos + 50, 50])

        # Add background parameters
        bg = np.min(spectrum)
        if background_order == 0:
            p0.append(bg)
            lower_bounds.append(0)
            upper_bounds.append(np.max(spectrum) * 0.5)
        elif background_order == 1:
            p0.extend([bg, 0])
            lower_bounds.extend([0, -0.1])
            upper_bounds.extend([np.max(spectrum) * 0.5, 0.1])

        # Fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    multi_gaussian,
                    bins,
                    spectrum,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=20000
                )

            # Extract results for each peak
            results = []
            n_peaks = len(peak_positions)

            for i in range(n_peaks):
                amp = popt[i * 3]
                mean = popt[i * 3 + 1]
                sigma = popt[i * 3 + 2]

                fwhm = 2.355 * sigma
                area = amp * sigma * np.sqrt(2 * np.pi)

                results.append({
                    'amplitude': float(amp),
                    'mean': float(mean),
                    'sigma': float(sigma),
                    'fwhm': float(fwhm),
                    'area': float(area),
                    'success': True
                })

            return results

        except Exception as e:
            warnings.warn(f"Multi-peak fit failed: {str(e)}")
            return [{'success': False, 'error': str(e)} for _ in peak_positions]
