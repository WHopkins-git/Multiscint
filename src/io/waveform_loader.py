"""
Waveform data loading and management
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
import json
from torch.utils.data import Dataset


@dataclass
class Waveform:
    """
    Container for a single waveform with metadata

    Attributes:
        waveform: Array of ADC samples
        timestamp: Event timestamp (μs)
        baseline: Baseline level (ADC counts)
        amplitude: Peak amplitude (ADC counts)
        energy: Calibrated energy (keV), if available
        scintillator: Scintillator type ('LYSO', 'BGO', 'NaI', 'Plastic')
        source: Radiation source ('Cs137', 'Na22', etc.)
        metadata: Additional metadata dictionary
    """
    waveform: np.ndarray
    timestamp: float
    baseline: float
    amplitude: float
    energy: Optional[float] = None
    scintillator: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict] = None

    @property
    def samples(self) -> int:
        """Number of samples in waveform"""
        return len(self.waveform)

    @property
    def baseline_corrected(self) -> np.ndarray:
        """Waveform with baseline subtracted"""
        return self.waveform - self.baseline

    @property
    def peak_index(self) -> int:
        """Index of peak sample"""
        return int(np.argmax(self.waveform))


class WaveformLoader:
    """
    Load waveform data from various formats

    Supports:
        - HDF5 files (preferred format)
        - NumPy binary files (.npy, .npz)
        - CSV files (for small datasets)

    Parameters:
        data_dir: Root directory containing data
        sampling_rate_MHz: DAQ sampling rate in MHz (default: 125 MS/s for DT5825S)

    Example:
        >>> loader = WaveformLoader("data/raw", sampling_rate_MHz=125)
        >>> waveforms = loader.load_waveforms("LYSO", "Cs137", n_waveforms=1000)
    """

    def __init__(self, data_dir: Union[str, Path], sampling_rate_MHz: float = 125.0):
        self.data_dir = Path(data_dir)
        self.sampling_rate_MHz = sampling_rate_MHz
        self.dt_ns = 1000.0 / sampling_rate_MHz  # Time per sample in nanoseconds

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_waveforms(
        self,
        scintillator: str,
        source: str,
        n_waveforms: Optional[int] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        run_number: Optional[int] = None
    ) -> List[Waveform]:
        """
        Load waveforms for a specific scintillator-source combination

        Parameters:
            scintillator: Scintillator type ('LYSO', 'BGO', 'NaI', 'Plastic')
            source: Radiation source ('Cs137', 'Na22', 'Co60', etc.)
            n_waveforms: Maximum number of waveforms to load (None = all)
            energy_range: Optional (E_min, E_max) filter in keV
            run_number: Specific run number to load (None = first available)

        Returns:
            List of Waveform objects
        """
        # Construct file path
        scint_dir = self.data_dir / scintillator / source

        if not scint_dir.exists():
            raise FileNotFoundError(f"Directory not found: {scint_dir}")

        # Find waveform files
        h5_files = list(scint_dir.glob("*.h5"))

        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {scint_dir}")

        # Select run
        if run_number is not None:
            h5_files = [f for f in h5_files if f"run{run_number:03d}" in f.name]
            if not h5_files:
                raise FileNotFoundError(f"Run {run_number} not found in {scint_dir}")

        file_path = h5_files[0]

        # Load from HDF5
        waveforms = self._load_from_hdf5(
            file_path,
            scintillator=scintillator,
            source=source,
            n_waveforms=n_waveforms,
            energy_range=energy_range
        )

        return waveforms

    def _load_from_hdf5(
        self,
        file_path: Path,
        scintillator: str,
        source: str,
        n_waveforms: Optional[int] = None,
        energy_range: Optional[Tuple[float, float]] = None
    ) -> List[Waveform]:
        """Load waveforms from HDF5 file"""
        waveforms = []

        with h5py.File(file_path, 'r') as f:
            # Load datasets
            waveform_data = f['waveforms'][:]
            timestamps = f['timestamps'][:]
            baselines = f['baselines'][:]

            # Optional: energies if calibrated
            if 'energies' in f:
                energies = f['energies'][:]
            else:
                energies = None

            # Load metadata
            metadata = {}
            if 'metadata' in f:
                for key, value in f['metadata'].attrs.items():
                    metadata[key] = value

            # Calculate amplitudes
            amplitudes = np.max(waveform_data, axis=1)

            # Apply energy filter if specified
            if energy_range is not None and energies is not None:
                E_min, E_max = energy_range
                mask = (energies >= E_min) & (energies <= E_max)
                indices = np.where(mask)[0]
            else:
                indices = np.arange(len(waveform_data))

            # Limit number of waveforms
            if n_waveforms is not None:
                indices = indices[:n_waveforms]

            # Create Waveform objects
            for idx in indices:
                waveforms.append(Waveform(
                    waveform=waveform_data[idx],
                    timestamp=timestamps[idx],
                    baseline=baselines[idx],
                    amplitude=amplitudes[idx],
                    energy=energies[idx] if energies is not None else None,
                    scintillator=scintillator,
                    source=source,
                    metadata=metadata.copy()
                ))

        return waveforms

    def get_available_data(self) -> pd.DataFrame:
        """
        Get summary of available data files

        Returns:
            DataFrame with columns: scintillator, source, run, n_events, file_path
        """
        data_summary = []

        for scint_dir in self.data_dir.iterdir():
            if not scint_dir.is_dir():
                continue

            scintillator = scint_dir.name

            for source_dir in scint_dir.iterdir():
                if not source_dir.is_dir():
                    continue

                source = source_dir.name

                for h5_file in source_dir.glob("*.h5"):
                    # Extract run number from filename
                    run_match = h5_file.stem

                    # Get number of events
                    with h5py.File(h5_file, 'r') as f:
                        n_events = len(f['waveforms'])

                    data_summary.append({
                        'scintillator': scintillator,
                        'source': source,
                        'run': run_match,
                        'n_events': n_events,
                        'file_path': str(h5_file)
                    })

        return pd.DataFrame(data_summary)


class WaveformDataset(Dataset):
    """
    PyTorch Dataset for waveforms

    Parameters:
        waveforms: List of Waveform objects
        transform: Optional transform function for waveforms
        normalize: Whether to normalize waveforms

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = WaveformDataset(waveforms, normalize=True)
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(
        self,
        waveforms: List[Waveform],
        transform=None,
        normalize: bool = True
    ):
        self.waveforms = waveforms
        self.transform = transform
        self.normalize = normalize

        # Create label encoding
        self.scintillators = sorted(list(set(w.scintillator for w in waveforms)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.scintillators)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict]:
        waveform_obj = self.waveforms[idx]

        # Get waveform (baseline corrected)
        waveform = waveform_obj.baseline_corrected.copy()

        # Normalize if requested
        if self.normalize and waveform_obj.amplitude > 0:
            waveform = waveform / waveform_obj.amplitude

        # Apply transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Get label
        label = self.label_to_idx[waveform_obj.scintillator]

        # Metadata
        metadata = {
            'amplitude': waveform_obj.amplitude,
            'baseline': waveform_obj.baseline,
            'timestamp': waveform_obj.timestamp,
            'energy': waveform_obj.energy,
            'scintillator': waveform_obj.scintillator,
            'source': waveform_obj.source
        }

        return waveform.astype(np.float32), label, metadata

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced datasets"""
        labels = [self.label_to_idx[w.scintillator] for w in self.waveforms]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        return weights
