"""SiPM saturation model"""

import numpy as np
from scipy.optimize import curve_fit


class SaturationModel:
    """Model SiPM saturation effects"""

    def __init__(self, n_cells: int = 18000):
        self.n_cells = n_cells

    def saturation_function(self, n_photons: np.ndarray, pde: float = 0.3) -> np.ndarray:
        """
        SiPM saturation model: N_fired = N_cells * (1 - exp(-N_photons*PDE/N_cells))

        Parameters:
            n_photons: Number of incident photons
            pde: Photon detection efficiency

        Returns:
            Number of fired cells
        """
        return self.n_cells * (1 - np.exp(-n_photons * pde / self.n_cells))

    def correct_saturation(self, measured_signal: np.ndarray, pde: float = 0.3) -> np.ndarray:
        """
        Apply inverse saturation correction

        Parameters:
            measured_signal: Measured signal (fired cells)
            pde: Photon detection efficiency

        Returns:
            Corrected photon count
        """
        # Inverse of saturation function
        n_photons = -self.n_cells / pde * np.log(1 - measured_signal / self.n_cells)
        return n_photons

    def fit_saturation_curve(self, photon_counts: np.ndarray, measured_signals: np.ndarray) -> Dict:
        """Fit saturation model to data"""
        try:
            def model(N, pde):
                return self.saturation_function(N, pde)

            popt, pcov = curve_fit(model, photon_counts, measured_signals, p0=[0.3], bounds=([0.01], [1.0]))

            return {
                'pde': popt[0],
                'pde_error': np.sqrt(pcov[0, 0]),
                'success': True
            }
        except:
            return {'success': False}
