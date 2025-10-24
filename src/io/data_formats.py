"""
Data format conversion and storage utilities
"""

import h5py
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Union
from .waveform_loader import Waveform


def save_hdf5(
    waveforms: np.ndarray,
    output_path: Union[str, Path],
    timestamps: np.ndarray = None,
    baselines: np.ndarray = None,
    energies: np.ndarray = None,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Save waveforms to HDF5 format

    Parameters:
        waveforms: Array of waveforms, shape (N, M) where N=events, M=samples
        output_path: Output file path
        timestamps: Event timestamps
        baselines: Baseline values for each waveform
        energies: Calibrated energies (if available)
        metadata: Dictionary of metadata to store as attributes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Store waveform data
        f.create_dataset('waveforms', data=waveforms, compression='gzip')

        if timestamps is not None:
            f.create_dataset('timestamps', data=timestamps, compression='gzip')

        if baselines is not None:
            f.create_dataset('baselines', data=baselines, compression='gzip')

        if energies is not None:
            f.create_dataset('energies', data=energies, compression='gzip')

        # Store metadata
        if metadata is not None:
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                meta_group.attrs[key] = value


def load_hdf5(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from HDF5 file

    Returns:
        Dictionary with 'waveforms', 'timestamps', 'baselines', 'energies', 'metadata'
    """
    file_path = Path(file_path)

    data = {}

    with h5py.File(file_path, 'r') as f:
        data['waveforms'] = f['waveforms'][:]

        if 'timestamps' in f:
            data['timestamps'] = f['timestamps'][:]

        if 'baselines' in f:
            data['baselines'] = f['baselines'][:]

        if 'energies' in f:
            data['energies'] = f['energies'][:]

        if 'metadata' in f:
            metadata = {}
            for key, value in f['metadata'].attrs.items():
                metadata[key] = value
            data['metadata'] = metadata

    return data


def save_processed_data(
    data: Union[pd.DataFrame, np.ndarray, Dict],
    output_path: Union[str, Path],
    format: str = 'auto'
) -> None:
    """
    Save processed data in appropriate format

    Parameters:
        data: Data to save (DataFrame, array, or dict)
        output_path: Output file path
        format: 'auto', 'csv', 'pickle', 'npy', 'json'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'auto':
        ext = output_path.suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext == '.pkl':
            format = 'pickle'
        elif ext == '.npy':
            format = 'npy'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}")

    if format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            raise TypeError("CSV format requires pandas DataFrame")

    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    elif format == 'npy':
        if isinstance(data, np.ndarray):
            np.save(output_path, data)
        else:
            raise TypeError("NPY format requires numpy array")

    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    else:
        raise ValueError(f"Unknown format: {format}")


def load_processed_data(file_path: Union[str, Path], format: str = 'auto') -> Any:
    """
    Load processed data from file

    Parameters:
        file_path: Path to file
        format: 'auto', 'csv', 'pickle', 'npy', 'json'

    Returns:
        Loaded data (type depends on format)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if format == 'auto':
        ext = file_path.suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext == '.pkl':
            format = 'pickle'
        elif ext == '.npy':
            format = 'npy'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}")

    if format == 'csv':
        return pd.read_csv(file_path)

    elif format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    elif format == 'npy':
        return np.load(file_path)

    elif format == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)

    else:
        raise ValueError(f"Unknown format: {format}")
