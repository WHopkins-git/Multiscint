"""
Parsers for N42 and CSV data formats from CAEN digitizers
"""

import numpy as np
from xml.etree import ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re


def import_n42_spectrum(filepath: str) -> Dict[str, Any]:
    """
    Parse ANSI N42.42 XML spectrum file from CAEN digitizer

    Parameters:
        filepath: Path to .n42 file

    Returns:
        Dictionary containing:
            - counts: List[int] - histogram counts per channel
            - start_time: str - ISO-8601 timestamp
            - stop_time: str - ISO-8601 timestamp
            - live_time: str - ISO-8601 duration
            - real_time: str - ISO-8601 duration
            - calibration: Dict - energy calibration if available
            - instrument: Dict - instrument metadata
            - n_channels: int - number of channels
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Define namespace (N42 often uses xmlns)
    ns = {'n42': 'http://physics.nist.gov/N42/2011/N42'}

    # Try without namespace first
    def find_elem(parent, tag):
        """Find element with or without namespace"""
        elem = parent.find(tag)
        if elem is None:
            elem = parent.find(f'{{*}}{tag}')
        if elem is None:
            elem = parent.find(f'n42:{tag}', ns)
        return elem

    def findall_elem(parent, tag):
        """Find all elements with or without namespace"""
        elems = parent.findall(tag)
        if not elems:
            elems = parent.findall(f'{{*}}{tag}')
        if not elems:
            elems = parent.findall(f'n42:{tag}', ns)
        return elems

    result = {
        'counts': [],
        'start_time': None,
        'stop_time': None,
        'live_time': None,
        'real_time': None,
        'calibration': None,
        'instrument': {},
        'n_channels': 0
    }

    # Extract instrument information
    inst_info = find_elem(root, 'RadInstrumentInformation')
    if inst_info is not None:
        manufacturer = find_elem(inst_info, 'RadInstrumentManufacturerName')
        model = find_elem(inst_info, 'RadInstrumentModelName')
        class_code = find_elem(inst_info, 'RadInstrumentClassCode')

        result['instrument']['manufacturer'] = manufacturer.text if manufacturer is not None else None
        result['instrument']['model'] = model.text if model is not None else None
        result['instrument']['class'] = class_code.text if class_code is not None else None

        # Version information
        versions = {}
        for version_elem in findall_elem(inst_info, 'RadInstrumentVersion'):
            comp_name = find_elem(version_elem, 'RadInstrumentComponentName')
            comp_version = find_elem(version_elem, 'RadInstrumentComponentVersion')
            if comp_name is not None and comp_version is not None:
                versions[comp_name.text] = comp_version.text
        result['instrument']['versions'] = versions

    # Extract measurement data
    rad_measurement = find_elem(root, 'RadMeasurement')
    if rad_measurement is not None:
        # Timing information
        start_time = find_elem(rad_measurement, 'StartDateTime')
        stop_time = find_elem(rad_measurement, 'StopDateTime')
        live_time = find_elem(rad_measurement, 'LiveTimeDuration')
        real_time = find_elem(rad_measurement, 'RealTimeDuration')

        result['start_time'] = start_time.text if start_time is not None else None
        result['stop_time'] = stop_time.text if stop_time is not None else None
        result['live_time'] = live_time.text if live_time is not None else None
        result['real_time'] = real_time.text if real_time is not None else None

        # Spectrum data
        spectrum = find_elem(rad_measurement, 'Spectrum')
        if spectrum is not None:
            channel_data = find_elem(spectrum, 'ChannelData')
            if channel_data is not None:
                # Split on any whitespace and convert to integers
                counts_text = channel_data.text.strip()
                counts = [int(x) for x in counts_text.split()]
                result['counts'] = counts
                result['n_channels'] = len(counts)

            # Try to get energy calibration reference
            cal_ref = spectrum.get('energyCalibrationReference')
            if cal_ref:
                # Find the calibration
                for cal_elem in findall_elem(root, 'EnergyCalibration'):
                    if cal_elem.get('id') == cal_ref:
                        calibration = {}

                        # Check for polynomial coefficients
                        coeffs = []
                        for coeff_elem in findall_elem(cal_elem, 'Coefficient'):
                            coeffs.append(float(coeff_elem.text))

                        if coeffs:
                            calibration['type'] = 'polynomial'
                            calibration['coefficients'] = coeffs

                        # Check for calibration points
                        points = []
                        for point_elem in findall_elem(cal_elem, 'EnergyCalibrationPoint'):
                            channel = find_elem(point_elem, 'Channel')
                            energy = find_elem(point_elem, 'EnergyValue')
                            if channel is not None and energy is not None:
                                unit = energy.get('unit', 'keV')
                                points.append({
                                    'channel': int(channel.text),
                                    'energy': float(energy.text),
                                    'unit': unit
                                })

                        if points:
                            calibration['type'] = 'points'
                            calibration['points'] = points

                        if calibration:
                            result['calibration'] = calibration

    return result


def import_waveforms_csv(filepath: str, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Parse semicolon-delimited CSV file with event waveforms from CAEN digitizer

    Format: BOARD;CHANNEL;TIMETAG;ENERGY;ENERGYSHORT;FLAGS;PROBE_CODE;SAMPLES

    Parameters:
        filepath: Path to .CSV file
        max_events: Maximum number of events to load (None = all)

    Returns:
        List of event dictionaries with:
            - board: int
            - channel: int
            - timetag: int (device ticks)
            - energy: int (long gate integration)
            - energy_short: int (short gate integration)
            - flags: str (hex string)
            - probe_code: int
            - samples: np.ndarray (ADC waveform)
    """
    events = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Read and validate header
        header_line = f.readline().strip()
        header = header_line.split(';')

        expected = ["BOARD", "CHANNEL", "TIMETAG", "ENERGY", "ENERGYSHORT", "FLAGS", "PROBE_CODE", "SAMPLES"]

        # Check header (case-insensitive)
        header_upper = [h.upper() for h in header[:len(expected)]]
        if header_upper != expected:
            raise ValueError(f"Unexpected header. Expected {expected}, got {header_upper}")

        # Read event rows
        for line_num, line in enumerate(f, start=2):
            if max_events is not None and len(events) >= max_events:
                break

            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                tokens = line.split(';')

                if len(tokens) < 8:
                    print(f"Warning: Line {line_num} has only {len(tokens)} fields, skipping")
                    continue

                # Parse scalar fields
                board = int(tokens[0])
                channel = int(tokens[1])
                timetag = int(tokens[2])
                energy = int(tokens[3])
                energy_short = int(tokens[4])
                flags = tokens[5]  # Keep as string (e.g., '0x0')
                probe_code = int(tokens[6])

                # Parse waveform (all remaining tokens)
                samples = np.array([int(x) for x in tokens[7:]], dtype=np.int32)

                events.append({
                    'board': board,
                    'channel': channel,
                    'timetag': timetag,
                    'energy': energy,
                    'energy_short': energy_short,
                    'flags': flags,
                    'probe_code': probe_code,
                    'samples': samples
                })

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue

    return events


def parse_iso8601_duration(duration_str: str) -> float:
    """
    Parse ISO 8601 duration string to seconds

    Example: 'PT0H6M16.547S' -> 376.547 seconds

    Parameters:
        duration_str: ISO 8601 duration string

    Returns:
        Duration in seconds
    """
    if not duration_str:
        return 0.0

    # Simple regex for PT#H#M#S format
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?'
    match = re.match(pattern, duration_str)

    if not match:
        raise ValueError(f"Cannot parse duration: {duration_str}")

    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def estimate_sampling_rate(waveforms: List[Dict[str, Any]], nominal_rate_MHz: float = 250.0) -> float:
    """
    Estimate sampling rate from waveform data

    For CAEN DT5720D, typical rate is 250 MS/s

    Parameters:
        waveforms: List of waveform dictionaries
        nominal_rate_MHz: Nominal sampling rate in MHz

    Returns:
        Estimated sampling rate in MHz
    """
    # For now, return nominal rate
    # Could be refined by analyzing timetag differences if needed
    return nominal_rate_MHz


def convert_csv_to_waveform_objects(events: List[Dict[str, Any]],
                                   scintillator: str,
                                   source: str,
                                   sampling_rate_MHz: float = 250.0):
    """
    Convert CSV event dictionaries to Waveform objects

    Parameters:
        events: List of event dictionaries from import_waveforms_csv
        scintillator: Scintillator type
        source: Radiation source
        sampling_rate_MHz: Digitizer sampling rate

    Returns:
        List of Waveform objects
    """
    from .waveform_loader import Waveform

    waveforms = []

    for event in events:
        # Calculate baseline from first 50 samples
        samples = event['samples']
        if len(samples) < 100:
            continue  # Skip too-short waveforms

        baseline = np.mean(samples[:50])
        amplitude = np.max(samples) - baseline

        # Convert timetag to microseconds (assuming 250 MS/s clock)
        timestamp_us = event['timetag'] / (sampling_rate_MHz * 1e6) * 1e6

        waveform = Waveform(
            waveform=samples.astype(np.float32),
            timestamp=timestamp_us,
            baseline=float(baseline),
            amplitude=float(amplitude),
            energy=float(event['energy']),  # Use digitizer energy estimate
            scintillator=scintillator,
            source=source,
            metadata={
                'board': event['board'],
                'channel': event['channel'],
                'timetag': event['timetag'],
                'energy_short': event['energy_short'],
                'flags': event['flags'],
                'probe_code': event['probe_code']
            }
        )

        waveforms.append(waveform)

    return waveforms
