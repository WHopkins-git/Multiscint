"""Comparative visualization functions"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi


def plot_comparison(data_dict, ylabel, title, **kwargs):
    """Bar chart comparison"""
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    plt.bar(labels, values)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    return plt.gcf()


def plot_radar_chart(categories, values_dict, title="Performance Comparison"):
    """Radar/spider chart for multi-dimensional comparison"""
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for label, values in values_dict.items():
        values = list(values) + [values[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim([0, 1])
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    return fig
