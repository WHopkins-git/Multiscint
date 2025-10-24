"""
Setup script for SiPM Scintillator Analysis Package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sipm-scintillator-analysis",
    version="1.0.0",
    author="SiPM Analysis Team",
    author_email="sipm-analysis@example.com",
    description="Comprehensive framework for SiPM-scintillator detector characterization with advanced ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sipm-scintillator-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "h5py>=3.1.0",
        "PyWavelets>=1.1.1",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "shap>=0.40.0",
        "pyyaml>=5.4.0",
        "joblib>=1.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "wavelet": [
            "kymatio>=0.3.0"
        ]
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json"]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "sipm-analyze=scripts.analyze:main",
        ],
    },
)
