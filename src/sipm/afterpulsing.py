"""Afterpulsing analysis for SiPMs"""

import numpy as np
from scipy.optimize import curve_fit


class AfterpulsingAnalyzer:
    """Analyze afterpulsing in SiPMs"""

    def analyze_inter_event_times(
        self,
        timestamps: np.ndarray,
        max_delay_us: float = 100
    ) -> Dict:
        """
        Analyze inter-event time distribution for afterpulsing

        Parameters:
            timestamps: Event timestamps (μs)
            max_delay_us: Maximum delay to consider

        Returns:
            Dictionary with afterpulsing probability and time constant
        """
        # Calculate inter-event times
        delta_t = np.diff(timestamps)

        # Filter to range of interest
        delta_t = delta_t[delta_t < max_delay_us]

        # Histogram
        hist, bins = np.histogram(delta_t, bins=np.logspace(-1, np.log10(max_delay_us), 100))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Fit exponential model
        def model(t, A_random, rate, A_afterpulse, tau):
            return A_random * np.exp(-rate * t) + A_afterpulse * np.exp(-t / tau)

        try:
            p0 = [np.max(hist), 0.01, 0.1 * np.max(hist), 10]
            popt, _ = curve_fit(model, bin_centers, hist, p0=p0, maxfev=5000)

            afterpulse_prob = popt[2] / (popt[0] + popt[2]) * 100
            tau_afterpulse = popt[3]

            return {
                'afterpulsing_probability': afterpulse_prob,
                'tau_afterpulsing': tau_afterpulse,
                'success': True
            }
        except:
            return {'success': False}
