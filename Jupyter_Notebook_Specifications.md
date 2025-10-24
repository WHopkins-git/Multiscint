# Jupyter Notebook Implementation Specifications

## Overview
This document provides detailed specifications for implementing 8 comprehensive Jupyter notebooks for SiPM-scintillator detector analysis. Each notebook is designed to be self-contained yet part of a coherent analysis pipeline.

---

## Notebook 1: Data Loading and Exploration
**Filename:** `01_data_loading_exploration.ipynb`

### Objectives
- Load waveform data from various formats (HDF5, NPY, CSV)
- Visualize raw pulse shapes
- Perform data quality checks
- Generate summary statistics

### Required Sections

#### 1.1 Setup and Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import seaborn as sns

# Local imports
from src.io import WaveformLoader
from src.visualization import plot_waveform, plot_waveform_grid
```

#### 1.2 Data Directory Structure
- Show directory tree
- List available scintillators and sources
- Count files per configuration

#### 1.3 Load Example Waveforms
```python
loader = WaveformLoader(data_dir="data/raw", sampling_rate_MHz=125)

# Load examples from each scintillator
lyso_pulses = loader.load_waveforms("LYSO", "Cs137", n_waveforms=100)
bgo_pulses = loader.load_waveforms("BGO", "Cs137", n_waveforms=100)
nai_pulses = loader.load_waveforms("NaI", "Cs137", n_waveforms=100)
plastic_pulses = loader.load_waveforms("Plastic", "Cs137", n_waveforms=100)
```

#### 1.4 Waveform Visualization
- Individual pulse plots with annotations
- Overlay of example pulses from each scintillator
- Time axis in nanoseconds
- Amplitude in ADC counts

#### 1.5 Statistical Summary
- Pulse amplitude distributions
- Baseline stability
- Trigger efficiency
- Data quality metrics

#### 1.6 Data Quality Checks
```python
def check_data_quality(pulses):
    """
    Comprehensive quality checks
    Returns: DataFrame with quality metrics
    """
    checks = {
        'n_pulses': len(pulses),
        'baseline_mean': np.mean([p.baseline for p in pulses]),
        'baseline_std': np.std([p.baseline for p in pulses]),
        'amplitude_range': [min, max],
        'saturation_count': count_saturated(pulses),
        'anomaly_count': count_anomalies(pulses)
    }
    return pd.DataFrame([checks])
```

#### 1.7 Interactive Exploration
- Widget to select scintillator and source
- Display random pulses
- Zoom capabilities

#### 1.8 Summary Report
- Table of loaded datasets
- Quality assessment
- Recommendations for next steps

### Expected Outputs
- 5-10 publication-quality figures
- Summary statistics table
- Data quality report
- Cleaned waveform dataset saved to processed/

### Estimated Runtime
- 5-10 minutes for loading and analysis
- Dataset size: ~1-10 GB

---

## Notebook 2: Energy Calibration
**Filename:** `02_energy_calibration.ipynb`

### Objectives
- Create energy spectra from pulse amplitudes
- Identify photopeaks for all sources
- Perform linear energy calibration
- Calculate energy resolution
- Compare scintillators

### Required Sections

#### 2.1 Setup
```python
from src.calibration import EnergyCalibrator, PeakFinder
from src.visualization import plot_spectrum, plot_calibration_curve
```

#### 2.2 Generate Raw Spectra
For each scintillator:
```python
# Extract pulse amplitudes
amplitudes = [pulse.amplitude for pulse in pulses]

# Create histogram (spectrum)
hist, bins = np.histogram(amplitudes, bins=1000, range=[0, max_adc])

# Plot with log scale
plot_spectrum(bins, hist, title=f"{scintillator} - {source}")
```

#### 2.3 Automated Peak Finding
```python
peak_finder = PeakFinder(min_prominence=0.05)
peaks = peak_finder.find_peaks(spectrum)

# Annotate peaks on spectrum
for peak in peaks:
    plt.axvline(peak['position'], color='red', linestyle='--')
    plt.text(peak['position'], peak['height'], f"{peak['position']:.0f}")
```

#### 2.4 Peak Fitting
For each identified peak:
```python
def fit_photopeak(spectrum, peak_position, fit_range=50):
    """
    Fit Gaussian + background to photopeak
    
    Returns:
        amplitude, mean, sigma, background
    """
    # Extract region around peak
    # Fit: A * exp(-(x-μ)²/2σ²) + B + C*x
    # Return fit parameters and FWHM
```

#### 2.5 Energy Calibration
```python
calibrator = EnergyCalibrator()

# Known peak energies (keV)
known_peaks = {
    'Am241': [59.5],
    'Co57': [122, 136],
    'Cs137': [662],
    'Na22': [511, 1275],
    'Co60': [1173, 1332]
}

# Calibrate
slope, intercept, r_squared = calibrator.calibrate(measured_peaks, known_peaks)

# Apply calibration
energy_spectrum = calibrator.apply_calibration(raw_spectrum, slope, intercept)
```

#### 2.6 Calibration Quality
- Plot calibration curve (measured vs. known energy)
- Show residuals
- Calculate R² and chi-square
- Check linearity

#### 2.7 Energy Resolution Calculation
```python
def calculate_resolution(peak_fit):
    """
    Calculate energy resolution from Gaussian fit
    
    Resolution (%) = (FWHM / Peak_position) * 100
    FWHM = 2.355 * sigma
    """
    fwhm = 2.355 * peak_fit['sigma']
    resolution = (fwhm / peak_fit['mean']) * 100
    return resolution
```

Plot resolution vs. energy for each scintillator

#### 2.8 Comparative Analysis
Create comparison table:

| Scintillator | Res @ 122 keV | Res @ 662 keV | Res @ 1332 keV | Linearity (R²) |
|--------------|---------------|---------------|----------------|----------------|
| LYSO | ... | ... | ... | ... |
| BGO | ... | ... | ... | ... |
| NaI | ... | ... | ... | ... |
| Plastic | ... | ... | ... | ... |

#### 2.9 Save Calibration Results
```python
# Save calibration parameters
calibration_results = {
    'LYSO': {'slope': ..., 'intercept': ..., 'r_squared': ...},
    'BGO': {...},
    'NaI': {...},
    'Plastic': {...}
}

with open('data/processed/energy_calibration.yaml', 'w') as f:
    yaml.dump(calibration_results, f)
```

### Expected Outputs
- 15-20 figures (spectra, calibration curves, resolution plots)
- Calibration parameters file
- Energy-calibrated spectra saved
- Comparison table

### Estimated Runtime
- 20-30 minutes

---

## Notebook 3: Pulse Shape Analysis
**Filename:** `03_pulse_shape_analysis.ipynb`

### Objectives
- Extract comprehensive pulse shape features
- Measure timing characteristics
- Fit decay curves
- Perform traditional pulse shape discrimination (PSD)
- Compare scintillator signatures

### Required Sections

#### 3.1 Setup
```python
from src.pulse_analysis import PulseFeatureExtractor, PulseFitter
from scipy.optimize import curve_fit
```

#### 3.2 Feature Extraction Pipeline
```python
feature_extractor = PulseFeatureExtractor(sampling_rate_MHz=125)

# Extract features for all pulses
feature_df = pd.DataFrame()

for pulse in pulses:
    features = feature_extractor.extract_features(pulse)
    feature_df = feature_df.append(features, ignore_index=True)

# Save feature database
feature_df.to_csv('data/processed/pulse_features.csv', index=False)
```

#### 3.3 Timing Analysis

**3.3.1 Rise Time Measurements**
```python
# Calculate 10-90% rise time for each scintillator
rise_times = {}
for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    pulses = load_pulses(scint)
    rise_times[scint] = [calculate_rise_time(p.waveform) for p in pulses]

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, scint in zip(axes.flat, rise_times.keys()):
    ax.hist(rise_times[scint], bins=50, alpha=0.7)
    ax.set_xlabel('Rise Time (ns)')
    ax.set_title(f'{scint} - Mean: {np.mean(rise_times[scint]):.2f} ns')
```

**3.3.2 Fall Time Measurements**
Similar analysis for fall time

**3.3.3 Peak Position Timing**
- Time from trigger to peak
- Jitter analysis

#### 3.4 Decay Curve Fitting

**3.4.1 Single Exponential Fit (BGO, NaI, Plastic)**
```python
def exponential_decay(t, A, tau, baseline):
    """Single exponential: A * exp(-t/tau) + baseline"""
    return A * np.exp(-t / tau) + baseline

# Fit decay
popt, pcov = curve_fit(exponential_decay, t_data, amplitude_data)
tau_fitted = popt[1]

# Plot fit quality
plt.plot(t_data, amplitude_data, 'o', label='Data')
plt.plot(t_data, exponential_decay(t_data, *popt), '-', label='Fit')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (ADC)')
plt.legend()
```

**3.4.2 Double Exponential Fit (LYSO)**
```python
def double_exponential(t, A1, tau1, A2, tau2, baseline):
    """Fast + slow component"""
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2) + baseline

# LYSO has ~40 ns (fast) + ~200 ns (slow) components
```

**3.4.3 Decay Time Summary Table**

| Scintillator | Literature τ (ns) | Measured τ (ns) | Fit Quality (R²) |
|--------------|-------------------|-----------------|------------------|
| Plastic | 2.4 | 2.3 ± 0.2 | 0.998 |
| LYSO | 40 (fast) | 38 ± 3 | 0.995 |
| NaI | 230 | 235 ± 10 | 0.997 |
| BGO | 300 | 305 ± 15 | 0.996 |

#### 3.5 Charge Integration Analysis

**3.5.1 Total Charge Calculation**
```python
def calculate_total_charge(waveform, baseline):
    """Integrate pulse above baseline"""
    waveform_corrected = waveform - baseline
    total_charge = np.sum(waveform_corrected[waveform_corrected > 0])
    return total_charge
```

**3.5.2 Tail Charge and PSD**
```python
def calculate_psd_parameter(waveform, baseline, tail_gate_ns=100):
    """
    Calculate tail-to-total charge ratio for PSD
    """
    # Find peak position
    peak_idx = np.argmax(waveform)
    
    # Define tail gate (samples after peak)
    tail_start = peak_idx + int(tail_gate_ns / dt_ns)
    
    # Calculate charges
    total_charge = calculate_total_charge(waveform, baseline)
    tail_charge = np.sum(waveform[tail_start:] - baseline)
    
    psd = tail_charge / total_charge if total_charge > 0 else 0
    return psd, tail_charge, total_charge
```

**3.5.3 PSD for Beta/Gamma Discrimination (Plastic Scintillator)**
```python
# Calculate PSD for Sr90 (beta) vs. gamma sources
psd_sr90 = [calculate_psd_parameter(p.waveform, p.baseline)[0] for p in sr90_pulses]
psd_cs137 = [calculate_psd_parameter(p.waveform, p.baseline)[0] for p in cs137_pulses]

# Plot 2D histogram: PSD vs. Energy
plt.hist2d(energies, psd_values, bins=100, cmap='viridis')
plt.xlabel('Energy (keV)')
plt.ylabel('PSD Parameter')
plt.colorbar(label='Counts')

# Calculate Figure of Merit (FOM)
fom = calculate_fom(psd_sr90, psd_cs137)
print(f"PSD Figure of Merit: {fom:.2f}")
```

#### 3.6 Shape Features

**3.6.1 Pulse Width (FWHM)**
```python
def calculate_fwhm(waveform, baseline):
    """Full Width at Half Maximum"""
    waveform_corrected = waveform - baseline
    peak = np.max(waveform_corrected)
    half_max = peak / 2
    
    # Find indices where waveform crosses half-max
    above_half = waveform_corrected >= half_max
    crossings = np.diff(above_half.astype(int))
    
    leading_edge = np.where(crossings == 1)[0][0]
    trailing_edge = np.where(crossings == -1)[0][0]
    
    fwhm_ns = (trailing_edge - leading_edge) * dt_ns
    return fwhm_ns
```

**3.6.2 Statistical Shape Descriptors**
- Skewness: Asymmetry of pulse
- Kurtosis: "Peakedness"
- Compare across scintillators

#### 3.7 Frequency Domain Analysis

```python
from scipy.fft import fft, fftfreq

def frequency_analysis(waveform, sampling_rate_MHz):
    """
    FFT of pulse waveform
    """
    # Compute FFT
    n = len(waveform)
    fft_values = fft(waveform)
    frequencies = fftfreq(n, 1/(sampling_rate_MHz * 1e6))
    
    # Plot power spectrum
    power = np.abs(fft_values)**2
    plt.plot(frequencies[:n//2], power[:n//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.xscale('log')
    plt.yscale('log')
    
    # Find dominant frequency
    dominant_freq = frequencies[np.argmax(power[1:n//2])]
    return dominant_freq, power
```

#### 3.8 Comparative Feature Visualization

**3.8.1 Feature Distributions**
Create grid of histograms showing feature distributions for each scintillator

**3.8.2 PCA Visualization**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_df)

# PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Plot
plt.figure(figsize=(10, 8))
for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    mask = feature_df['scintillator'] == scint
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                label=scint, alpha=0.5, s=10)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.title('PCA of Pulse Shape Features')
```

**3.8.3 Correlation Matrix**
Heatmap showing correlation between features

#### 3.9 Summary Statistics Table

Generate comprehensive table:
- Mean and std for each feature
- Per scintillator
- Save to CSV/LaTeX

### Expected Outputs
- 25-30 figures
- Feature database (CSV)
- Decay time measurements
- PSD analysis results

### Estimated Runtime
- 30-45 minutes

---

## Notebook 4: Machine Learning Classification
**Filename:** `04_ml_classification.ipynb`

### Objectives
- Prepare ML dataset
- Train multiple classification models
- Evaluate and compare performance
- Interpret model predictions
- Demonstrate real-time classification

### Required Sections

#### 4.1 Setup and Data Preparation
```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
```

#### 4.2 Load Feature Dataset
```python
# Load features extracted in Notebook 3
features_df = pd.read_csv('data/processed/pulse_features.csv')

# Select features for ML
feature_columns = [
    'amplitude', 'rise_time_10_90', 'fall_time_90_10',
    'total_charge', 'tail_total_ratio', 'width_fwhm',
    'skewness', 'kurtosis', 'dominant_frequency',
    'decay_constant', 'baseline_std'
]

X = features_df[feature_columns].values
y = features_df['scintillator'].values  # Labels: 'LYSO', 'BGO', 'NaI', 'Plastic'

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y_encoded)}")
```

#### 4.3 Train-Val-Test Split
```python
# Split: 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

#### 4.4 Baseline: Random Forest
```python
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, classes=['LYSO', 'BGO', 'NaI', 'Plastic'])

# Feature importance
importances = rf.feature_importances_
plot_feature_importance(feature_columns, importances)
```

#### 4.5 XGBoost Classifier
```python
# XGBoost with hyperparameter tuning
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 9]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb*100:.2f}%")
print(f"Best parameters: {grid_search.best_params_}")
```

#### 4.6 Support Vector Machine
```python
# SVM with RBF kernel
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm*100:.2f}%")
```

#### 4.7 Neural Network (MLP)
```python
# Multi-layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {acc_mlp*100:.2f}%")
```

#### 4.8 Model Comparison
```python
# Create comparison table
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'SVM', 'MLP'],
    'Accuracy': [acc_rf, acc_xgb, acc_svm, acc_mlp],
    'Precision': [...],  # Calculate per model
    'Recall': [...],
    'F1-Score': [...]
})

print(results.sort_values('Accuracy', ascending=False))

# Bar plot comparison
plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['Accuracy'] * 100)
plt.ylabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.ylim([85, 100])
```

#### 4.9 Deep Learning: CNN on Raw Waveforms

**4.9.1 Prepare Waveform Dataset**
```python
# Load raw waveforms (not features)
def load_waveform_dataset(scintillators, n_per_class=10000):
    X_waves = []
    y_waves = []
    
    for i, scint in enumerate(scintillators):
        pulses = load_pulses(scint, n=n_per_class)
        for p in pulses:
            waveform_normalized = (p.waveform - p.baseline) / p.amplitude
            X_waves.append(waveform_normalized)
            y_waves.append(i)
    
    X_waves = np.array(X_waves).reshape(-1, 1024, 1)  # (N, length, channels)
    y_waves = np.array(y_waves)
    return X_waves, y_waves

X_cnn, y_cnn = load_waveform_dataset(['LYSO', 'BGO', 'NaI', 'Plastic'])

# Split
X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
)
```

**4.9.2 Build CNN Architecture**
```python
def build_cnn_model(input_shape=(1024, 1), num_classes=4):
    model = keras.Sequential([
        keras.layers.Conv1D(32, kernel_size=16, strides=2, activation='relu', 
                           input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        
        keras.layers.Conv1D(64, kernel_size=8, strides=2, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        keras.layers.Conv1D(128, kernel_size=4, strides=1, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

cnn_model = build_cnn_model()
cnn_model.summary()
```

**4.9.3 Train CNN**
```python
# Callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train
history = cnn_model.fit(
    X_cnn_train, y_cnn_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```

**4.9.4 Evaluate CNN**
```python
# Test set performance
y_pred_cnn = np.argmax(cnn_model.predict(X_cnn_test), axis=1)
acc_cnn = accuracy_score(y_cnn_test, y_pred_cnn)
print(f"CNN Accuracy: {acc_cnn*100:.2f}%")

# Confusion matrix
cm_cnn = confusion_matrix(y_cnn_test, y_pred_cnn)
plot_confusion_matrix(cm_cnn, classes=['LYSO', 'BGO', 'NaI', 'Plastic'])

# Classification report
print(classification_report(y_cnn_test, y_pred_cnn, 
                          target_names=['LYSO', 'BGO', 'NaI', 'Plastic']))
```

#### 4.10 Robustness Testing

**4.10.1 Energy Dependence**
```python
# Test performance across different energy bins
energy_bins = [(0, 200), (200, 500), (500, 800), (800, 1500)]

for e_min, e_max in energy_bins:
    mask = (features_df['energy'] >= e_min) & (features_df['energy'] < e_max)
    X_bin = X_test_scaled[mask]
    y_bin = y_test[mask]
    
    accuracy = best_xgb.score(X_bin, y_bin)
    print(f"Energy {e_min}-{e_max} keV: Accuracy = {accuracy*100:.2f}%")
```

**4.10.2 Noise Robustness**
```python
# Add synthetic noise and test degradation
noise_levels = [0, 0.05, 0.1, 0.2, 0.5]
accuracies = []

for noise in noise_levels:
    X_noisy = X_test_scaled + np.random.normal(0, noise, X_test_scaled.shape)
    acc = best_xgb.score(X_noisy, y_test)
    accuracies.append(acc)

plt.plot(noise_levels, accuracies, 'o-')
plt.xlabel('Noise Level (std)')
plt.ylabel('Accuracy')
plt.title('Model Robustness to Noise')
```

**4.10.3 Cross-Source Generalization**
```python
# Train on some sources, test on others
train_sources = ['Cs137', 'Co60', 'Na22']
test_sources = ['Co57', 'Am241']

# Filter dataset and retrain
# Evaluate on held-out sources
```

#### 4.11 Model Interpretability

**4.11.1 Feature Importance (XGBoost)**
```python
import shap

# SHAP values
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_columns)

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0], 
                feature_names=feature_columns)
```

**4.11.2 CNN Visualization (Grad-CAM alternative)**
```python
# Saliency map: which parts of waveform most important?
def compute_saliency(model, waveform):
    """
    Compute gradient of output w.r.t. input
    """
    waveform_tensor = tf.convert_to_tensor(waveform.reshape(1, 1024, 1))
    with tf.GradientTape() as tape:
        tape.watch(waveform_tensor)
        predictions = model(waveform_tensor)
        predicted_class = tf.argmax(predictions, axis=1)
        class_score = predictions[:, predicted_class]
    
    gradient = tape.gradient(class_score, waveform_tensor)
    return gradient.numpy().squeeze()

# Plot example saliency map
example_waveform = X_cnn_test[0]
saliency = compute_saliency(cnn_model, example_waveform)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(example_waveform.squeeze())
plt.title('Waveform')

plt.subplot(2, 1, 2)
plt.plot(np.abs(saliency))
plt.title('Saliency (Importance)')
plt.xlabel('Time Sample')
```

#### 4.12 Real-Time Classification Demo
```python
# Simulate real-time classification
def classify_unknown_pulse(waveform):
    """
    Given a raw waveform, identify scintillator
    """
    # Extract features
    features = feature_extractor.extract_features_from_waveform(waveform)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    prediction = best_xgb.predict(features_scaled)[0]
    probabilities = best_xgb.predict_proba(features_scaled)[0]
    
    scintillator = le.inverse_transform([prediction])[0]
    confidence = probabilities[prediction] * 100
    
    print(f"Predicted: {scintillator} (Confidence: {confidence:.1f}%)")
    return scintillator, confidence

# Test on random pulses
for i in range(5):
    random_pulse = select_random_pulse()
    classify_unknown_pulse(random_pulse.waveform)
```

#### 4.13 Model Deployment
```python
# Save best models
import joblib

joblib.dump(best_xgb, 'results/models/xgboost_classifier.pkl')
joblib.dump(scaler, 'results/models/feature_scaler.pkl')
joblib.dump(le, 'results/models/label_encoder.pkl')

cnn_model.save('results/models/cnn_classifier.h5')

print("Models saved for deployment!")
```

#### 4.14 Summary and Recommendations
- Best model selection
- Performance summary table
- Deployment considerations
- Future improvements

### Expected Outputs
- 30-40 figures
- Trained models saved
- Performance metrics table
- SHAP plots and interpretability

### Estimated Runtime
- Feature-based ML: 10-20 minutes
- CNN training: 1-4 hours (GPU), 12-24 hours (CPU)

---

## Notebook 5: Pile-up Detection and Correction
**Filename:** `05_pileup_correction.ipynb`

### Objectives
- Characterize pile-up at different count rates
- Implement detection algorithms
- Develop correction methods
- Validate improvements

### Required Sections

#### 5.1 Setup
```python
from src.pileup import PileupDetector, PileupCorrector
from scipy.signal import find_peaks
```

#### 5.2 Pile-up Simulation
```python
def simulate_pileup(pulse1, pulse2, delay_ns, sampling_ns=8):
    """
    Create synthetic pile-up by overlaying two pulses
    """
    delay_samples = int(delay_ns / sampling_ns)
    combined = pulse1.copy()
    
    if delay_samples < len(pulse1):
        overlap_length = len(pulse1) - delay_samples
        combined[delay_samples:] += pulse2[:overlap_length]
    
    return combined

# Generate pile-up examples with varying delays
delays = [10, 50, 100, 200, 500]  # ns
for delay in delays:
    pileup_waveform = simulate_pileup(pulse1, pulse2, delay)
    plot_waveform(pileup_waveform, title=f"Pile-up with {delay} ns delay")
```

#### 5.3 Count Rate Measurements
```python
# Measure actual count rate for each scintillator-source combination
def calculate_count_rate(timestamps, measurement_time_s):
    """
    Calculate true and measured count rates
    Account for dead time
    """
    n_events = len(timestamps)
    measured_rate = n_events / measurement_time_s
    
    # Estimate dead time
    inter_event_times = np.diff(timestamps)
    dead_time_fraction = np.sum(inter_event_times < min_inter_event_time) / n_events
    
    # Correct for dead time
    true_rate = measured_rate / (1 - dead_time_fraction)
    
    return measured_rate, true_rate, dead_time_fraction

# Apply to all datasets
rate_summary = {}
for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    for source in ['Cs137', 'Co60', ...]:
        timestamps = load_timestamps(scint, source)
        rates = calculate_count_rate(timestamps, measurement_time)
        rate_summary[(scint, source)] = rates
```

#### 5.4 Pile-up Detection Methods

**5.4.1 Baseline Restoration Method**
```python
def detect_pileup_baseline(waveform, baseline, threshold_samples=50):
    """
    Check if baseline restored before next trigger
    """
    # Baseline should be restored in tail
    tail = waveform[-threshold_samples:]
    baseline_restored = np.abs(np.mean(tail) - baseline) < 3 * np.std(tail)
    
    return not baseline_restored  # True if pile-up
```

**5.4.2 Pulse Shape Fitting Method**
```python
def detect_pileup_fit(waveform, decay_time_ns):
    """
    Fit expected pulse shape and check residuals
    """
    fitted_pulse = fit_exponential_decay(waveform, decay_time_ns)
    residuals = waveform - fitted_pulse
    chi_square = np.sum(residuals**2) / len(waveform)
    
    # High chi-square indicates poor fit (pile-up)
    threshold = calculate_threshold(decay_time_ns)
    return chi_square > threshold
```

**5.4.3 Second Derivative Method**
```python
def detect_pileup_derivative(waveform):
    """
    Look for anomalies in second derivative
    """
    d2_waveform = np.diff(waveform, n=2)
    anomaly_score = np.max(np.abs(d2_waveform))
    
    threshold = 3 * np.std(d2_waveform)
    return anomaly_score > threshold
```

**5.4.4 ML-Based Detection**
```python
# Train binary classifier for pile-up detection
# Features: fit quality, shape parameters, baseline stats
# Labels: clean (0) vs. pile-up (1)

pileup_classifier = RandomForestClassifier(...)
pileup_classifier.fit(X_train_pileup, y_train_pileup)

# Use in detection pipeline
```

#### 5.5 Method Comparison
```python
# Compare detection methods on simulated data
methods = {
    'Baseline': detect_pileup_baseline,
    'Fit Quality': detect_pileup_fit,
    'Derivative': detect_pileup_derivative,
    'ML': pileup_classifier.predict
}

# Generate test set: 50% clean, 50% pile-up
test_pulses = generate_test_set(n_clean=1000, n_pileup=1000)

results = {}
for name, method in methods.items():
    predictions = [method(pulse) for pulse in test_pulses]
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}

pd.DataFrame(results).T
```

#### 5.6 Pile-up Correction: Deconvolution

**5.6.1 Algorithm Implementation**
```python
def deconvolve_pileup(waveform, decay_time_ns, sampling_ns=8):
    """
    Separate two overlapping pulses
    """
    # 1. Identify first pulse peak
    peak1_idx = np.argmax(waveform)
    peak1_amplitude = waveform[peak1_idx]
    
    # 2. Model first pulse (exponential from peak)
    t = np.arange(len(waveform) - peak1_idx) * sampling_ns
    model_pulse1 = peak1_amplitude * np.exp(-t / decay_time_ns)
    
    # 3. Subtract from waveform
    residual = waveform.copy()
    residual[peak1_idx:] -= model_pulse1
    
    # 4. Check for second pulse
    threshold = 5 * np.std(residual[:peak1_idx])  # Noise level
    if np.max(residual) > threshold:
        peak2_idx = np.argmax(residual)
        peak2_amplitude = residual[peak2_idx]
        
        # Estimate energies
        energy1 = np.sum(model_pulse1)
        energy2 = np.sum(residual[residual > 0])
        
        return True, energy1, energy2, peak2_idx - peak1_idx
    else:
        # False alarm, only one pulse
        return False, np.sum(waveform), 0, 0
```

**5.6.2 Validation on Simulated Data**
```python
# Test deconvolution accuracy
true_energies1 = []
true_energies2 = []
recovered_energies1 = []
recovered_energies2 = []

for i in range(1000):
    pulse1, pulse2 = generate_random_pulses()
    delay = np.random.uniform(50, 300)  # ns
    pileup = simulate_pileup(pulse1, pulse2, delay)
    
    success, e1, e2, delay_recovered = deconvolve_pileup(pileup, decay_time)
    
    if success:
        true_energies1.append(get_energy(pulse1))
        true_energies2.append(get_energy(pulse2))
        recovered_energies1.append(e1)
        recovered_energies2.append(e2)

# Plot recovery accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(true_energies1, recovered_energies1, alpha=0.5)
plt.plot([0, max(true_energies1)], [0, max(true_energies1)], 'r--')
plt.xlabel('True Energy (ADC)')
plt.ylabel('Recovered Energy (ADC)')
plt.title('First Pulse Recovery')

plt.subplot(1, 2, 2)
plt.scatter(true_energies2, recovered_energies2, alpha=0.5)
plt.plot([0, max(true_energies2)], [0, max(true_energies2)], 'r--')
plt.xlabel('True Energy (ADC)')
plt.ylabel('Recovered Energy (ADC)')
plt.title('Second Pulse Recovery')
```

#### 5.7 Scintillator-Specific Analysis

**Compare pile-up effects across scintillators:**

| Scintillator | Decay (ns) | Pile-up % @ 10k cps | Pile-up % @ 50k cps | Deconv Success Rate |
|--------------|------------|---------------------|---------------------|---------------------|
| Plastic | 2.4 | <1% | 2% | 95% |
| LYSO | 40 | 2% | 8% | 85% |
| NaI | 230 | 10% | 35% | 60% |
| BGO | 300 | 15% | 45% | 55% |

#### 5.8 Digital Filter Optimization

**Trapezoidal shaping parameter sweep:**

```python
def test_shaping_parameters(waveforms, rise_times, flat_times):
    """
    Test different digital filter parameters
    Measure: throughput, resolution, pile-up rejection
    """
    results = []
    
    for rise in rise_times:
        for flat in flat_times:
            filter_output = apply_trapezoidal_filter(waveforms, rise, flat)
            
            # Metrics
            throughput = calculate_throughput(filter_output)
            resolution = calculate_resolution(filter_output)
            pileup_rate = calculate_pileup_rate(filter_output)
            
            results.append({
                'rise_ns': rise,
                'flat_ns': flat,
                'throughput': throughput,
                'resolution': resolution,
                'pileup_rate': pileup_rate
            })
    
    return pd.DataFrame(results)

# Optimize for each scintillator
optimal_params = {}
for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    results = test_shaping_parameters(...)
    optimal = results.loc[results['throughput'].idxmax()]
    optimal_params[scint] = optimal
```

#### 5.9 Spectrum Correction
```python
def correct_spectrum(spectrum_raw, pileup_fraction):
    """
    Apply statistical correction to spectrum
    """
    # First-order correction
    spectrum_corrected = spectrum_raw / (1 - pileup_fraction)
    
    # Peak position restoration
    # (if pile-up shifts peaks)
    
    return spectrum_corrected

# Apply and compare
spectrum_raw = load_spectrum('LYSO', 'Cs137', high_rate=True)
spectrum_corrected = correct_spectrum(spectrum_raw, pileup_fraction=0.08)

plt.figure()
plt.plot(energy_axis, spectrum_raw, label='Raw (with pile-up)')
plt.plot(energy_axis, spectrum_corrected, label='Corrected')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.legend()
```

#### 5.10 Dead Time Calculation
```python
def calculate_dead_time(inter_event_times, min_processing_time_ns):
    """
    Estimate dead time from inter-event time distribution
    """
    dead_events = np.sum(inter_event_times < min_processing_time_ns)
    total_events = len(inter_event_times)
    dead_time_fraction = dead_events / total_events
    
    return dead_time_fraction * 100  # Percent
```

#### 5.11 Performance Summary

Create comprehensive comparison:
- Raw vs. corrected spectra
- Energy resolution preservation
- Throughput improvement
- Dead time analysis

### Expected Outputs
- 20-25 figures
- Pile-up detection algorithm comparison
- Deconvolution validation
- Optimized filter parameters

### Estimated Runtime
- 30-45 minutes

---

## Notebook 6: SiPM Characterization
**Filename:** `06_sipm_characterization.ipynb`

### Objectives
- Measure crosstalk probability
- Characterize afterpulsing
- Quantify saturation effects
- Assess impact on performance

### Required Sections

#### 6.1 Setup
```python
from src.sipm import CrosstalkAnalyzer, AfterpulsingAnalyzer, SaturationModel
```

#### 6.2 Single Photoelectron Spectrum
```python
# Use very low light (weak source or LED at low intensity)
def measure_spe_spectrum(pulses_low_light):
    """
    Identify discrete photoelectron peaks
    """
    amplitudes = [p.amplitude for p in pulses_low_light]
    hist, bins = np.histogram(amplitudes, bins=500, range=[0, max(amplitudes)])
    
    # Fit multiple Gaussians (0, 1, 2, ... p.e.)
    peaks = find_peaks(hist, min_height=0.1*max(hist))
    
    # Distance between peaks = single p.e. gain
    spe_gain = np.mean(np.diff([bins[p] for p in peaks]))
    
    plt.figure()
    plt.plot(bins[:-1], hist)
    for p in peaks:
        plt.axvline(bins[p], color='r', linestyle='--')
    plt.xlabel('Amplitude (ADC)')
    plt.ylabel('Counts')
    plt.title('Single Photoelectron Spectrum')
    
    return spe_gain, peaks
```

#### 6.3 Crosstalk Measurement

**6.3.1 Method 1: SPE Spectrum Analysis**
```python
def calculate_crosstalk_spe(spe_spectrum):
    """
    Calculate crosstalk from shoulder on 1 p.e. peak
    """
    # Fit 1 p.e. peak
    peak_1pe = fit_gaussian(spe_spectrum, peak_position=gain)
    
    # Measure excess counts above 1.5 p.e.
    threshold = 1.5 * gain
    excess_counts = np.sum(spe_spectrum[bins > threshold])
    total_1pe_counts = peak_1pe['area']
    
    crosstalk_prob = excess_counts / total_1pe_counts
    return crosstalk_prob * 100  # Percent
```

**6.3.2 Method 2: Energy-Dependent Crosstalk**
```python
def measure_crosstalk_vs_energy(pulses):
    """
    Measure crosstalk as function of primary photon count
    """
    # Bin pulses by amplitude (proxy for photon count)
    amplitude_bins = np.linspace(min_amp, max_amp, 20)
    crosstalk_vs_amplitude = []
    
    for i in range(len(amplitude_bins)-1):
        mask = (amplitudes >= amplitude_bins[i]) & (amplitudes < amplitude_bins[i+1])
        pulses_bin = pulses[mask]
        
        # Measure effective gain
        mean_charge = np.mean([p.total_charge for p in pulses_bin])
        expected_charge = amplitude_bins[i] * base_gain
        excess = (mean_charge - expected_charge) / expected_charge
        
        crosstalk_vs_amplitude.append(excess * 100)
    
    plt.plot(amplitude_bins[:-1], crosstalk_vs_amplitude, 'o-')
    plt.xlabel('Pulse Amplitude (ADC)')
    plt.ylabel('Crosstalk (%)')
    plt.title('Crosstalk vs. Signal Amplitude')
    
    return crosstalk_vs_amplitude
```

**6.3.3 Scintillator Comparison**
```python
# Compare crosstalk for different scintillators
crosstalk_results = {}

for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    pulses = load_pulses(scint, 'Cs137')
    crosstalk_prob = calculate_crosstalk_spe(pulses)
    crosstalk_vs_E = measure_crosstalk_vs_energy(pulses)
    
    crosstalk_results[scint] = {
        'average': crosstalk_prob,
        'vs_energy': crosstalk_vs_E
    }

# Plot comparison
plt.figure(figsize=(10, 6))
for scint, data in crosstalk_results.items():
    plt.plot(energy_axis, data['vs_energy'], 'o-', label=scint)
plt.xlabel('Energy (keV)')
plt.ylabel('Crosstalk Probability (%)')
plt.legend()
plt.title('Crosstalk Comparison Across Scintillators')
```

#### 6.4 Afterpulsing Analysis

**6.4.1 Inter-Event Time Distribution**
```python
def analyze_afterpulsing(timestamps):
    """
    Measure afterpulsing from inter-event time histogram
    """
    # Calculate time differences
    delta_t = np.diff(timestamps) * 1e6  # Convert to microseconds
    
    # Histogram (log scale)
    hist, bins = np.histogram(delta_t, bins=np.logspace(-1, 3, 100))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Expected: exponential (random) + exponential (afterpulsing)
    def model(t, A_random, rate_random, A_afterpulse, tau_afterpulse):
        return A_random * np.exp(-rate_random * t) + A_afterpulse * np.exp(-t / tau_afterpulse)
    
    # Fit
    popt, pcov = curve_fit(model, bin_centers, hist, p0=[max(hist), 0.01, 0.1*max(hist), 10])
    
    # Extract afterpulsing probability
    afterpulsing_prob = popt[2] / (popt[0] + popt[2]) * 100
    tau_afterpulse = popt[3]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(bin_centers, hist, 'o', label='Data')
    plt.semilogx(bin_centers, model(bin_centers, *popt), '-', label='Fit')
    plt.xlabel('Inter-Event Time (μs)')
    plt.ylabel('Counts')
    plt.title(f'Afterpulsing Analysis\nProbability: {afterpulsing_prob:.2f}%, τ: {tau_afterpulse:.1f} μs')
    plt.legend()
    
    return afterpulsing_prob, tau_afterpulse
```

**6.4.2 Amplitude Correlation**
```python
def correlate_afterpulsing_with_amplitude(pulses, timestamps):
    """
    Check if afterpulsing rate increases with pulse size
    """
    # For each pulse, check if afterpulse within next 100 μs
    amplitudes = [p.amplitude for p in pulses]
    has_afterpulse = []
    
    for i, timestamp in enumerate(timestamps[:-1]):
        next_time = timestamps[i+1] - timestamp
        if next_time < 100:  # Within 100 μs
            has_afterpulse.append(1)
        else:
            has_afterpulse.append(0)
    
    # Bin by amplitude and calculate afterpulse rate
    amplitude_bins = np.linspace(min(amplitudes), max(amplitudes), 10)
    afterpulse_rate_vs_amp = []
    
    for i in range(len(amplitude_bins)-1):
        mask = (amplitudes >= amplitude_bins[i]) & (amplitudes < amplitude_bins[i+1])
        rate = np.mean(np.array(has_afterpulse)[mask]) * 100
        afterpulse_rate_vs_amp.append(rate)
    
    plt.plot(amplitude_bins[:-1], afterpulse_rate_vs_amp, 'o-')
    plt.xlabel('Pulse Amplitude (ADC)')
    plt.ylabel('Afterpulsing Rate (%)')
    plt.title('Afterpulsing vs. Pulse Amplitude')
```

#### 6.5 Saturation Analysis

**6.5.1 Saturation Model Fitting**
```python
def fit_saturation_model(measured_amplitudes, photon_counts, n_cells):
    """
    Fit SiPM saturation model:
    N_fired = N_cells * (1 - exp(-N_photons/N_cells))
    """
    def saturation_model(N_photons, N_cells, pde):
        return N_cells * (1 - np.exp(-N_photons * pde / N_cells))
    
    # Fit
    popt, pcov = curve_fit(saturation_model, photon_counts, measured_amplitudes,
                           p0=[n_cells, 0.3])
    
    n_cells_fitted = popt[0]
    pde_fitted = popt[1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(photon_counts, measured_amplitudes, 'o', label='Data')
    plt.plot(photon_counts, saturation_model(photon_counts, *popt), '-', 
             label=f'Fit: N_cells={n_cells_fitted:.0f}, PDE={pde_fitted:.2f}')
    plt.xlabel('Estimated Photon Count')
    plt.ylabel('Measured Signal (Fired Cells)')
    plt.legend()
    plt.title('SiPM Saturation Curve')
    
    return n_cells_fitted, pde_fitted
```

**6.5.2 Energy-Dependent Saturation**
```python
# Use multi-energy sources
energies_kev = [59.5, 122, 511, 662, 1173, 1332]
sources = ['Am241', 'Co57', 'Na22', 'Cs137', 'Co60', 'Co60']

for scint in ['LYSO', 'NaI']:  # High light yield scintillators
    measured_peaks = []
    expected_peaks = []
    
    for energy, source in zip(energies_kev, sources):
        spectrum = load_calibrated_spectrum(scint, source)
        peak_position = find_photopeak(spectrum, energy)
        measured_peaks.append(peak_position)
        expected_peaks.append(energy)
    
    # Check linearity
    deviation = (np.array(measured_peaks) - np.array(expected_peaks)) / np.array(expected_peaks) * 100
    
    plt.figure()
    plt.plot(expected_peaks, deviation, 'o-', label=scint)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Deviation from Linearity (%)')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'{scint} Saturation Assessment')
    plt.legend()
```

#### 6.6 Photon Detection Efficiency (PDE)
```python
def estimate_pde(scintillator_light_yield, measured_signal, energy_kev):
    """
    Estimate PDE from measured signal
    """
    expected_photons = scintillator_light_yield * energy_kev / 1000  # ph/MeV * MeV
    measured_photoelectrons = measured_signal / spe_gain
    
    pde = measured_photoelectrons / expected_photons
    return pde

# Calculate for each scintillator
pde_results = {}
for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    light_yield = SCINTILLATORS[scint].light_yield
    measured = measure_signal_at_662kev(scint)
    pde = estimate_pde(light_yield, measured, 662)
    pde_results[scint] = pde

print(pd.DataFrame.from_dict(pde_results, orient='index', columns=['PDE']))
```

#### 6.7 Temperature Dependence (Optional)
```python
# If temperature-controlled measurements available
def analyze_temperature_dependence(measurements):
    """
    Plot key parameters vs. temperature:
    - Breakdown voltage
    - Gain
    - Dark count rate
    - Crosstalk
    """
    temperatures = [m['temperature'] for m in measurements]
    breakdown_voltages = [m['v_breakdown'] for m in measurements]
    gains = [m['gain'] for m in measurements]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(temperatures, breakdown_voltages, 'o-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Breakdown Voltage (V)')
    
    plt.subplot(1, 3, 2)
    plt.plot(temperatures, gains, 'o-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Gain (ADC/p.e.)')
    
    plt.subplot(1, 3, 3)
    plt.plot(temperatures, [m['dark_rate'] for m in measurements], 'o-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Dark Count Rate (Hz)')
    plt.yscale('log')
```

#### 6.8 Impact on Energy Resolution
```python
def quantify_sipm_effects_on_resolution(scintillator):
    """
    Decompose energy resolution into components:
    - Statistical fluctuations (photon)
    - Scintillator (light yield, non-proportionality)
    - SiPM (crosstalk, afterpulsing, noise)
    - Electronics
    """
    # Measure resolution with ideal vs. real SiPM
    # Compare to theoretical limit
    
    theoretical_resolution = calculate_theoretical_resolution(scintillator)
    measured_resolution = measure_resolution_662kev(scintillator)
    
    excess = measured_resolution - theoretical_resolution
    
    print(f"{scintillator}:")
    print(f"  Theoretical: {theoretical_resolution:.1f}%")
    print(f"  Measured: {measured_resolution:.1f}%")
    print(f"  SiPM contribution: {excess:.1f}%")
```

#### 6.9 Summary Tables and Recommendations

**Table 1: SiPM Characterization Summary**

| Scintillator | Crosstalk (%) | Afterpulsing (%) | Saturation Energy (keV) | PDE (%) | Resolution Degradation (%) |
|--------------|---------------|------------------|-------------------------|---------|---------------------------|
| LYSO | ... | ... | ... | ... | ... |
| BGO | ... | ... | ... | ... | ... |
| NaI | ... | ... | ... | ... | ... |
| Plastic | ... | ... | ... | ... | ... |

**Recommendations:**
- Overvoltage optimization per scintillator
- When to worry about saturation
- Temperature stabilization importance

### Expected Outputs
- 20-25 figures
- SiPM parameter measurements
- Saturation curves
- Impact assessment

### Estimated Runtime
- 30-45 minutes

---

## Notebook 7: Comprehensive Comparison and Visualization
**Filename:** `07_comprehensive_comparison.ipynb`

### Objectives
- Synthesize all analysis results
- Create publication-quality comparative figures
- Generate performance matrices
- Provide application-specific recommendations

### Required Sections

#### 7.1 Import All Results
```python
# Load results from previous notebooks
energy_calibration = pd.read_csv('data/processed/energy_calibration.csv')
pulse_features = pd.read_csv('data/processed/pulse_features.csv')
ml_results = joblib.load('results/ml_performance.pkl')
sipm_characterization = pd.read_csv('results/sipm_characterization.csv')
```

#### 7.2 Energy Performance Comparison

**7.2.1 Resolution vs. Energy**
```python
fig, ax = plt.subplots(figsize=(10, 6))

for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    resolutions = get_resolutions(scint)
    energies = [122, 511, 662, 1173, 1332]
    ax.plot(energies, resolutions, 'o-', label=scint, linewidth=2)

ax.set_xlabel('Energy (keV)', fontsize=14)
ax.set_ylabel('Energy Resolution (%)', fontsize=14)
ax.set_title('Energy Resolution Comparison', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

**7.2.2 Calibration Quality**
```python
# Side-by-side calibration curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, scint in zip(axes.flat, ['LYSO', 'BGO', 'NaI', 'Plastic']):
    plot_calibration_curve(ax, scint)
    ax.set_title(f'{scint} (R²={get_r_squared(scint):.4f})')
plt.tight_layout()
```

#### 7.3 Timing Performance Comparison

**7.3.1 Decay Time Comparison**
```python
decay_times = {
    'LYSO': 40,
    'BGO': 300,
    'NaI': 230,
    'Plastic': 2.4
}

plt.figure(figsize=(8, 6))
plt.barh(list(decay_times.keys()), list(decay_times.values()))
plt.xlabel('Decay Time (ns)', fontsize=14)
plt.title('Scintillator Decay Time Comparison', fontsize=16)
plt.xscale('log')
for i, (scint, tau) in enumerate(decay_times.items()):
    plt.text(tau, i, f' {tau} ns', va='center', fontsize=12)
```

**7.3.2 Representative Pulse Shapes**
```python
# Normalized overlay of typical pulses
fig, ax = plt.subplots(figsize=(12, 6))

for scint in ['LYSO', 'BGO', 'NaI', 'Plastic']:
    typical_pulse = get_average_pulse(scint, energy=662)
    normalized = (typical_pulse - np.min(typical_pulse)) / np.max(typical_pulse)
    time_axis = np.arange(len(normalized)) * 8  # ns
    ax.plot(time_axis, normalized, label=scint, linewidth=2)

ax.set_xlabel('Time (ns)', fontsize=14)
ax.set_ylabel('Normalized Amplitude', fontsize=14)
ax.set_title('Typical Pulse Shapes @ 662 keV', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim([0, 2000])
ax.grid(True, alpha=0.3)
```

#### 7.4 ML Classification Results

**7.4.1 Confusion Matrix Grid**
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
models = ['Random Forest', 'XGBoost', 'SVM', 'CNN']
cms = [cm_rf, cm_xgb, cm_svm, cm_cnn]

for ax, model, cm in zip(axes.flat, models, cms):
    plot_confusion_matrix(ax, cm, classes=['LYSO', 'BGO', 'NaI', 'Plastic'])
    ax.set_title(f'{model} (Acc: {get_accuracy(model):.1f}%)')

plt.tight_layout()
```

**7.4.2 Model Performance Bar Chart**
```python
accuracies = [acc_rf, acc_xgb, acc_svm, acc_mlp, acc_cnn]
models = ['Random\nForest', 'XGBoost', 'SVM', 'MLP', 'CNN']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, [a*100 for a in accuracies], color=['blue', 'green', 'red', 'orange', 'purple'])
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('ML Model Comparison for Scintillator Classification', fontsize=16)
plt.ylim([85, 100])

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=12)
```

#### 7.5 SiPM Effects Summary

**7.5.1 Crosstalk Comparison**
```python
crosstalk_data = get_crosstalk_summary()

plt.figure(figsize=(10, 6))
x = np.arange(len(crosstalk_data))
width = 0.35

plt.bar(x - width/2, crosstalk_data['low_amplitude'], width, label='Low Amplitude')
plt.bar(x + width/2, crosstalk_data['high_amplitude'], width, label='High Amplitude')

plt.xlabel('Scintillator', fontsize=14)
plt.ylabel('Crosstalk Probability (%)', fontsize=14)
plt.title('Crosstalk Comparison Across Scintillators', fontsize=16)
plt.xticks(x, ['LYSO', 'BGO', 'NaI', 'Plastic'])
plt.legend(fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
```

**7.5.2 Saturation Assessment**
```python
# Show which scintillators saturate SiPM
photons_per_kev = {
    'LYSO': 32,
    'BGO': 8.2,
    'NaI': 38,
    'Plastic': 10
}

saturation_threshold = 15000  # photons (example)

fig, ax = plt.subplots(figsize=(10, 6))
energies = np.linspace(0, 1500, 100)

for scint, phe_per_kev in photons_per_kev.items():
    photons = energies * phe_per_kev
    ax.plot(energies, photons, label=scint, linewidth=2)

ax.axhline(saturation_threshold, color='red', linestyle='--', linewidth=2, label='Saturation Threshold')
ax.set_xlabel('Energy (keV)', fontsize=14)
ax.set_ylabel('Photon Count', fontsize=14)
ax.set_title('SiPM Saturation Risk by Scintillator', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
```

#### 7.6 Multi-Dimensional Radar Chart
```python
from math import pi

def create_radar_chart(scintillators, metrics):
    """
    Create radar/spider chart for multi-dimensional comparison
    """
    categories = list(metrics[scintillators[0]].keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for scint in scintillators:
        values = list(metrics[scint].values())
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=scint)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim([0, 1])
    ax.set_title('Scintillator Performance Comparison', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)
    
    return fig

# Normalize all metrics to 0-1 scale
normalized_metrics = {
    'LYSO': {
        'Energy\nResolution': normalize(resolution_lyso, reverse=True),
        'Light\nYield': normalize(light_yield_lyso),
        'Timing': normalize(timing_lyso),
        'Stopping\nPower': normalize(stopping_power_lyso),
        'SiPM\nCompatibility': normalize(sipm_compat_lyso)
    },
    # ... similar for BGO, NaI, Plastic
}

radar_fig = create_radar_chart(['LYSO', 'BGO', 'NaI', 'Plastic'], normalized_metrics)
```

#### 7.7 Performance Matrix Heatmap
```python
# Create comprehensive performance matrix
performance_matrix = pd.DataFrame({
    'Energy Resolution @ 662 keV (%)': [10.5, 13.2, 7.1, 26.3],
    'Light Yield (ph/MeV/1000)': [32, 8.2, 38, 10],
    'Decay Time (ns)': [40, 300, 230, 2.4],
    'Density (g/cm³)': [7.1, 7.13, 3.67, 1.03],
    'Crosstalk (%)': [22, 12, 25, 15],
    'ML Accuracy (%)': [96.2, 94.8, 97.1, 95.5],
    'Cost (relative)': [4, 3, 2, 1]
}, index=['LYSO', 'BGO', 'NaI', 'Plastic'])

# Normalize columns for heatmap (0-1 scale)
normalized_matrix = (performance_matrix - performance_matrix.min()) / (performance_matrix.max() - performance_matrix.min())

# Some metrics are "better when lower" - reverse these
for col in ['Energy Resolution @ 662 keV (%)', 'Decay Time (ns)', 'Crosstalk (%)', 'Cost (relative)']:
    normalized_matrix[col] = 1 - normalized_matrix[col]

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(normalized_matrix.T, annot=performance_matrix.T, fmt='.1f', 
            cmap='RdYlGn', center=0.5, cbar_kws={'label': 'Normalized Performance'},
            linewidths=0.5)
plt.title('Scintillator Performance Matrix', fontsize=16)
plt.xlabel('Scintillator', fontsize=14)
plt.ylabel('Performance Metric', fontsize=14)
plt.tight_layout()
```

#### 7.8 Application-Specific Recommendations

**7.8.1 Decision Tree Visualization**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create simple decision tree based on application requirements
# Features: requires_timing, requires_resolution, requires_high_energy, cost_sensitive

def recommend_scintillator(requires_timing, requires_resolution, requires_high_energy, cost_sensitive):
    """
    Rule-based recommendation system
    """
    if requires_timing:
        if requires_high_energy:
            return 'LYSO'
        else:
            return 'Plastic'
    elif requires_resolution:
        if cost_sensitive:
            return 'NaI'
        else:
            return 'LYSO'
    elif requires_high_energy:
        return 'BGO'
    elif cost_sensitive:
        return 'Plastic'
    else:
        return 'NaI'

# Visualize decision tree
# (Could train actual decision tree classifier on application features)
```

**7.8.2 Application Suitability Table**

```python
applications = {
    'PET Imaging': {'LYSO': '★★★★★', 'BGO': '★★★☆☆', 'NaI': '★★☆☆☆', 'Plastic': '★☆☆☆☆'},
    'Gamma Spectroscopy': {'LYSO': '★★★★☆', 'BGO': '★★★☆☆', 'NaI': '★★★★★', 'Plastic': '★☆☆☆☆'},
    'High Energy Physics': {'LYSO': '★★★★☆', 'BGO': '★★★★★', 'NaI': '★★★☆☆', 'Plastic': '★★☆☆☆'},
    'Homeland Security': {'LYSO': '★★★☆☆', 'BGO': '★★★★☆', 'NaI': '★★★★☆', 'Plastic': '★★★★☆'},
    'Particle Detection': {'LYSO': '★★☆☆☆', 'BGO': '★☆☆☆☆', 'NaI': '★☆☆☆☆', 'Plastic': '★★★★★'},
    'Portable Detectors': {'LYSO': '★★★☆☆', 'BGO': '★★★★☆', 'NaI': '★★☆☆☆', 'Plastic': '★★★★★'}
}

app_df = pd.DataFrame(applications).T
print(app_df)
```

#### 7.9 Cost-Performance Trade-off
```python
# Bubble chart: Performance vs. Cost
fig, ax = plt.subplots(figsize=(10, 8))

scintillators = ['LYSO', 'BGO', 'NaI', 'Plastic']
performance_score = [85, 70, 90, 60]  # Composite score
cost_relative = [4, 3, 2, 1]  # Relative cost
popularity = [500, 300, 600, 400]  # Bubble size (citations, usage)

colors = ['blue', 'green', 'red', 'orange']

for scint, perf, cost, pop, color in zip(scintillators, performance_score, cost_relative, popularity, colors):
    ax.scatter(cost, perf, s=pop, alpha=0.6, c=color, edgecolors='black', linewidth=2)
    ax.text(cost, perf, f' {scint}', fontsize=14, va='center')

ax.set_xlabel('Relative Cost', fontsize=14)
ax.set_ylabel('Performance Score', fontsize=14)
ax.set_title('Cost-Performance Trade-off', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])
ax.set_ylim([50, 100])
```

#### 7.10 Summary Conclusions

**Generate final summary:**
1. Best overall performer
2. Best value (cost/performance)
3. Application-specific winners
4. Key trade-offs
5. Novel findings from this study

### Expected Outputs
- 15-20 publication-quality figures
- Comprehensive comparison tables
- Decision support tools
- Executive summary

### Estimated Runtime
- 20-30 minutes

---

## Notebook 8: Paper Figures Generation
**Filename:** `08_paper_figures.ipynb`

### Objectives
- Generate all figures for papers
- Ensure consistent styling
- High-resolution exports
- Supplementary materials

### Required Sections

#### 8.1 Style Configuration
```python
# Set publication-quality matplotlib parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5),  # Standard column width
    'lines.linewidth': 1.5,
    'lines.markersize': 6
})

# Color palette (colorblind-friendly)
colors = {
    'LYSO': '#0173B2',
    'BGO': '#DE8F05',
    'NaI': '#029E73',
    'Plastic': '#CC78BC'
}
```

#### 8.2 Paper 1 Figures (Comprehensive Comparison)

**Figure 1: Experimental Setup**
- Schematic diagram
- Photo of detector
- Block diagram of electronics

**Figure 2: Energy Spectra**
- 4-panel: One spectrum per scintillator (Cs-137)
- Annotated photopeaks

**Figure 3: Energy Calibration**
- Calibration curves
- Residual plots

**Figure 4: Energy Resolution**
- Resolution vs. energy
- Comparison to literature

**Figure 5: Pulse Shapes**
- Overlay of normalized typical pulses
- Inset: Early time detail

**Figure 6: Decay Time Measurements**
- Fitted decay curves
- Summary table

**Figure 7: SiPM Effects**
- Crosstalk vs. amplitude
- Saturation curves

**Figure 8: Performance Matrix**
- Heatmap summary

#### 8.3 Paper 2 Figures (ML Classification)

**Figure 1: Feature Distributions**
- Grid showing key features per scintillator

**Figure 2: PCA Visualization**
- 2D scatter of first two principal components

**Figure 3: Model Comparison**
- Bar chart of accuracies

**Figure 4: Confusion Matrices**
- Best model (XGBoost or CNN)

**Figure 5: Feature Importance**
- SHAP summary plot

**Figure 6: CNN Architecture**
- Diagram of network

**Figure 7: Learning Curves**
- Training/validation accuracy and loss

**Figure 8: Robustness**
- Accuracy vs. energy
- Accuracy vs. noise

#### 8.4 Paper 3 Figures (Pile-up Correction)

**Figure 1: Pile-up Examples**
- Simulated and real pile-up waveforms

**Figure 2: Detection Methods**
- ROC curves for different methods

**Figure 3: Deconvolution**
- Before/after examples
- Energy recovery accuracy

**Figure 4: Spectra Correction**
- Raw vs. corrected spectra

**Figure 5: Rate Dependence**
- Resolution vs. count rate
- Throughput improvement

#### 8.5 Supplementary Materials

- High-resolution versions of all figures
- Additional spectra for all sources
- Extended data tables
- Code snippets for key algorithms

#### 8.6 Export Functions
```python
def export_figure(fig, name, formats=['pdf', 'png', 'svg']):
    """
    Export figure in multiple formats for publication
    """
    for fmt in formats:
        filename = f'results/figures/paper/{name}.{fmt}'
        fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

# Export all paper figures
for fig_name, fig_object in paper_figures.items():
    export_figure(fig_object, fig_name)
```

#### 8.7 LaTeX Table Generation
```python
def dataframe_to_latex(df, caption, label):
    """
    Convert DataFrame to LaTeX table
    """
    latex_str = df.to_latex(
        index=True,
        escape=False,
        column_format='l' + 'c' * len(df.columns),
        caption=caption,
        label=label
    )
    
    # Save to file
    with open(f'results/tables/{label}.tex', 'w') as f:
        f.write(latex_str)
    
    return latex_str

# Generate all tables
energy_resolution_table = create_energy_resolution_table()
latex_table = dataframe_to_latex(
    energy_resolution_table,
    caption='Energy resolution comparison across scintillators',
    label='tab:energy_resolution'
)
```

### Expected Outputs
- 20-30 publication-ready figures (PDF, PNG, SVG)
- LaTeX tables
- Supplementary material package

### Estimated Runtime
- 15-20 minutes

---

## Implementation Notes for Claude Code

### General Guidelines

1. **Code Quality**
   - Follow PEP 8 style guide
   - Include comprehensive docstrings
   - Add type hints where appropriate
   - Include error handling

2. **Modularity**
   - Each notebook should be self-contained
   - Import from src/ modules
   - Avoid code duplication

3. **Reproducibility**
   - Set random seeds (np.random.seed(42))
   - Save/load intermediate results
   - Version control all code

4. **Performance**
   - Use vectorized operations (NumPy)
   - Parallelize where possible (multiprocessing)
   - Profile slow sections

5. **Visualization**
   - Clear axis labels and titles
   - Legends when multiple series
   - Consistent color schemes
   - Annotations for key features

6. **Documentation**
   - Markdown cells explaining each section
   - Comments in code for complex logic
   - References to literature where appropriate

### Data Format Assumptions

**Waveform Files:**
- HDF5 or NumPy arrays
- Shape: (n_events, n_samples)
- Metadata: scintillator, source, timestamp

**Processed Results:**
- CSV for tabular data
- Pickle or joblib for Python objects
- JSON for metadata

### Testing Strategy

- Unit tests for each function in src/
- Integration tests on small datasets
- Validation against known results

### Dependencies Management

```bash
# Create environment
conda create -n sipm-analysis python=3.10
conda activate sipm-analysis

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
```

---

**END OF NOTEBOOK SPECIFICATIONS**

These specifications provide complete implementation details for Claude Code to generate comprehensive, publication-quality Jupyter notebooks for the SiPM detector characterization study.