"""
SiPM Scintillator Analysis Package

A comprehensive framework for characterizing scintillation detectors coupled
to Silicon Photomultipliers using advanced digital pulse processing and
machine learning techniques.

Modules:
    io: Data loading and file I/O
    calibration: Energy calibration and spectroscopy
    pulse_analysis: Pulse shape feature extraction
    ml: Machine learning models (traditional and advanced)
    sipm: SiPM characterization (crosstalk, afterpulsing, saturation)
    pileup: Pile-up detection and correction
    visualization: Plotting and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "SiPM Analysis Team"

from . import io
from . import calibration
from . import pulse_analysis
from . import ml
from . import sipm
from . import pileup
from . import visualization

__all__ = [
    'io',
    'calibration',
    'pulse_analysis',
    'ml',
    'sipm',
    'pileup',
    'visualization'
]
