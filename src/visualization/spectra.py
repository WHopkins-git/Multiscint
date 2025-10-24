"""Spectroscopy plotting functions"""

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(bins, counts, title="Gamma Spectrum", xlabel="Energy (keV)", ylabel="Counts", **kwargs):
    """Plot gamma spectrum"""
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    plt.plot(bins[:-1], counts, drawstyle='steps-pre', **kwargs)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.yscale(kwargs.get('yscale', 'log'))
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_waveform(waveform, sampling_rate_MHz=125, title="Pulse Waveform", **kwargs):
    """Plot single waveform"""
    t = np.arange(len(waveform)) * (1000 / sampling_rate_MHz)  # Time in ns
    plt.figure(figsize=kwargs.get('figsize', (10, 4)))
    plt.plot(t, waveform, **kwargs)
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("Amplitude (ADC)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_waveform_grid(waveforms, labels=None, sampling_rate_MHz=125, **kwargs):
    """Plot grid of waveforms"""
    n = len(waveforms)
    cols = kwargs.get('cols', 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=kwargs.get('figsize', (12, 3 * rows)))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, waveform in enumerate(waveforms):
        t = np.arange(len(waveform)) * (1000 / sampling_rate_MHz)
        axes[idx].plot(t, waveform)
        axes[idx].set_xlabel("Time (ns)")
        axes[idx].set_ylabel("Amplitude")
        if labels:
            axes[idx].set_title(labels[idx])
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
