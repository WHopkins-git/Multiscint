# Comprehensive Characterization of SiPM-Coupled Scintillation Detectors: A Machine Learning and Advanced Signal Processing Approach

## Research Study Protocol v1.0
**Date:** October 2025  
**Detector System:** SiPM with interchangeable scintillators (LYSO, BGO, NaI, Plastic)  
**Digitizer:** CAEN DT5825S (125 MS/s, 14-bit)  
**Radiation Sources:** Cs-137, Na-22, Co-60, Co-57, Am-241, Sr-90

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scientific Background & Motivation](#scientific-background)
3. [Research Objectives](#research-objectives)
4. [Experimental Setup](#experimental-setup)
5. [Data Collection Protocol](#data-collection-protocol)
6. [Analysis Methods](#analysis-methods)
7. [Expected Results](#expected-results)
8. [Publication Strategy](#publication-strategy)
9. [Software Implementation Plan](#software-implementation-plan)
10. [Timeline & Milestones](#timeline)

---

## 1. Executive Summary

This study presents a comprehensive characterization of four scintillator materials (LYSO, BGO, NaI(Tl), and plastic) coupled to Silicon Photomultipliers (SiPMs) using modern digital pulse processing techniques. The research addresses critical gaps in the literature by:

1. **Systematic comparison** of scintillator-SiPM combinations using identical readout electronics
2. **Machine learning-based pulse shape discrimination** for automated scintillator identification
3. **Advanced pile-up correction algorithms** optimized for each scintillator's decay characteristics
4. **Comprehensive SiPM-specific characterization** including crosstalk and afterpulsing effects

**Novel Contributions:**
- First ML-based classification system for scintillator identification from raw pulse shapes
- Quantitative comparison of digital pulse processing parameters across scintillator types
- Systematic study of SiPM nonlinear effects with different light yield scintillators
- Open-source analysis toolkit for the radiation detection community

**Potential Impact:** 
- 2-3 peer-reviewed publications in journals like *IEEE Transactions on Nuclear Science*, *Nuclear Instruments and Methods A*
- Practical guidance for detector selection in medical imaging, homeland security, and high-energy physics
- Open-source software framework benefiting research community

---

## 2. Scientific Background & Motivation

### 2.1 Current State of the Field

**Silicon Photomultipliers (SiPMs)** have revolutionized radiation detection by offering:
- Compact size and low operating voltage compared to PMTs
- Insensitivity to magnetic fields (crucial for PET/MRI)
- Single photon sensitivity
- Fast timing characteristics

However, SiPMs introduce unique challenges:
- **Limited dynamic range** due to finite number of microcells
- **Crosstalk**: avalanche photons trigger adjacent cells
- **Afterpulsing**: trapped carriers cause delayed avalanches
- **Temperature sensitivity** of breakdown voltage

### 2.2 Research Gaps

**Gap 1: Lack of Systematic Comparisons**
- Most literature studies single scintillator-SiPM combinations
- No comprehensive dataset comparing multiple scintillators with identical SiPM and electronics
- Limited guidance on optimal scintillator selection for specific applications

**Gap 2: Underutilized Digital Signal Processing**
- Modern digitizers enable sophisticated pulse shape analysis
- Most studies use traditional analog electronics or simple digital methods
- Machine learning potential largely unexplored in scintillation detection

**Gap 3: SiPM Nonlinearity Not Well Characterized Across Scintillators**
- High light yield scintillators (NaI, LYSO) may saturate SiPM
- Crosstalk and afterpulsing effects vary with scintillator properties
- Limited quantitative data on these effects

**Gap 4: Pile-up Correction for Mixed Decay Times**
- Fast scintillators (LYSO) vs slow (BGO) require different approaches
- Digital pile-up correction algorithms rarely optimized per scintillator

### 2.3 Why This Study Matters

**Scientific Impact:**
- Provides benchmark dataset for scintillator-SiPM performance
- Advances understanding of SiPM behavior with varying light yields
- Demonstrates ML applicability to nuclear instrumentation

**Practical Impact:**
- Helps researchers select optimal detector configuration
- Provides validated analysis methods for pulse processing
- Reduces trial-and-error in detector development

---

## 3. Research Objectives

### 3.1 Primary Objectives

**Objective 1: Comprehensive Performance Characterization**
Quantify and compare the four scintillator-SiPM configurations across:
- Energy resolution at multiple gamma energies (59.5 - 1332 keV)
- Detection efficiency and photopeak-to-Compton ratios
- Timing characteristics (rise/fall times, decay constants)
- Low-energy detection thresholds
- Background and intrinsic radioactivity

**Objective 2: Machine Learning-Based Pulse Shape Discrimination**
Develop and validate ML models that can:
- Identify scintillator type from raw pulse waveforms alone
- Achieve >95% classification accuracy
- Operate in real-time (<1 ms per pulse)
- Generalize across different radiation energies

**Objective 3: SiPM Nonlinearity and Crosstalk Characterization**
Quantify SiPM-specific effects for each scintillator:
- Crosstalk probability vs. primary photon count
- Afterpulsing rates and time constants
- Saturation effects at high light yields
- Impact on energy resolution and linearity

**Objective 4: Advanced Pile-up Correction**
Develop scintillator-specific pile-up correction algorithms:
- Automatic pile-up detection
- Pulse deconvolution for overlapping events
- Comparison with traditional methods
- Dead time correction

### 3.2 Secondary Objectives

- Optimize digital filter parameters for each scintillator
- Create comprehensive pulse template library
- Investigate temperature dependence (if equipment available)
- Assess performance at different count rates

### 3.3 Success Criteria

✅ **Minimum Viable Study:**
- Complete energy calibration for all 4 scintillators × 6 sources
- Extract >1000 pulses per configuration for ML training
- Achieve ML classification accuracy >90%
- Quantify crosstalk in at least 2 scintillators
- Demonstrate pile-up correction improvement >20%

✅ **Ideal Complete Study:**
- ML accuracy >95% with cross-validation
- Full crosstalk characterization across energy range
- Temperature-dependent measurements
- Real-time implementation demonstration
- 3 publishable manuscripts

---

## 4. Experimental Setup

### 4.1 Detector Configuration

**SiPM Specifications:**
- Type: [To be specified - e.g., SensL MicroFB-30035-SMT]
- Active area: [e.g., 3×3 mm²]
- Microcell count: [e.g., ~18,000]
- Breakdown voltage: [e.g., 24.5V at 25°C]
- Overvoltage setting: [e.g., 3-5V above breakdown]
- Temperature: [Ambient or controlled]

**Scintillator Crystals:**

| Property | LYSO:Ce | BGO | NaI(Tl) | Plastic (BC-408) |
|----------|---------|-----|---------|------------------|
| Dimensions | TBD | TBD | TBD | TBD |
| Density (g/cm³) | 7.1 | 7.13 | 3.67 | 1.032 |
| Light Yield (ph/MeV) | 32,000 | 8,200 | 38,000 | 10,000 |
| Decay Time (ns) | 40 | 300 | 230 | 2.4 |
| Rise Time (ns) | 0.1 | 2 | 1 | 0.5 |
| Peak λ (nm) | 420 | 480 | 415 | 425 |
| Hygroscopic | No | No | Yes | No |
| Intrinsic Activity | Yes (Lu-176) | No | Yes (K-40) | No |

**Optical Coupling:**
- Coupling method: [e.g., optical grease, optical pad]
- Reflector material: [e.g., Teflon tape, ESR film]
- Light collection: Direct coupling or light guide?

### 4.2 Data Acquisition System

**CAEN DT5825S Specifications:**
- Channels: 8 (using 1 for this study)
- Sampling rate: 125 MS/s (8 ns per sample)
- Resolution: 14-bit (16384 ADC channels)
- Input range: 0.5 or 2.0 Vpp (to be specified)
- Trigger: Self-trigger with adjustable threshold
- Record length: [e.g., 1024 samples = 8.192 μs]

**Digital Pulse Processing Settings:**
- Trigger threshold: [To be optimized per scintillator]
- Baseline samples: [e.g., 100 samples before trigger]
- Energy filter: [Trapezoidal shaping parameters TBD]
- Pile-up rejection: [Enabled/Disabled for different studies]

**Data Storage:**
- Format: HDF5 for efficient storage and access
- Raw waveforms: Full ADC trace for each event
- Metadata: Timestamp, baseline, computed energy
- Estimated size: ~1-10 GB per scintillator-source combination

### 4.3 Radiation Sources

**Calibration Source Specifications:**

| Source | Half-life | Activity | Gamma Energies (keV) | Primary Use |
|--------|-----------|----------|----------------------|-------------|
| **Cs-137** | 30.17 y | [TBD μCi] | 662 | Primary calibration reference |
| **Na-22** | 2.6 y | [TBD μCi] | 511, 1275 | High/low energy calibration, timing |
| **Co-60** | 5.27 y | [TBD μCi] | 1173, 1332 | High energy, double peak resolution |
| **Co-57** | 271.8 d | [TBD μCi] | 122, 136 | Low energy performance |
| **Am-241** | 432.2 y | [TBD μCi] | 59.5 | Low energy threshold, X-rays |
| **Sr-90** | 28.8 y | [TBD μCi] | β⁻ (546 keV max) | Beta spectrum, plastic scintillator |

**Source Positioning:**
- Distance: [e.g., 10 cm from detector face]
- Geometry: Point source approximation
- Shielding: [Lead collimator if needed]
- Background: Measure without source for baseline

### 4.4 Environmental Control

**Temperature:**
- Ambient: [Record and monitor]
- Target: 20-25°C
- Variation: ±2°C during measurements
- Optional: Temperature-dependent study (-10 to +40°C)

**Humidity:** 
- Important for NaI (hygroscopic)
- Keep NaI sealed or in dry environment

**Electromagnetic Interference:**
- Shielded environment preferred
- Ground loops checked
- Cable routing to minimize noise

### 4.5 Measurement Protocols

**Standard Measurement:**
1. Detector warm-up: 30 minutes
2. Background measurement: 1 hour (no source)
3. Source measurement: Until 10,000-100,000 counts
4. Repeat for all source-scintillator combinations

**Extended Measurements (for pile-up studies):**
- Variable source distances to change count rate
- Target rates: 100, 1k, 10k, 50k cps
- Record true and measured count rates

**Special Studies:**
- **LED pulsing:** For SiPM characterization without radiation
- **Coincidence timing:** If dual detector available (Na-22)
- **Temperature sweeps:** If climate chamber available

---

## 5. Data Collection Protocol

### 5.1 Data Acquisition Strategy

**Phase 1: Initial Survey (Week 1)**
- Quick spectrum collection for all combinations
- Verify proper operation
- Identify optimal trigger thresholds
- Estimate measurement times needed

**Phase 2: Comprehensive Calibration (Weeks 2-3)**
- High-statistics runs (100k+ events per configuration)
- 4 scintillators × 6 sources = 24 datasets
- Include background measurements
- Multiple runs per configuration for reproducibility

**Phase 3: Waveform Collection (Weeks 3-4)**
- Raw waveform storage for ML training
- Target: 10,000-50,000 pulses per scintillator
- Stratified sampling across energy ranges
- Include "difficult" cases (pile-up, low amplitude)

**Phase 4: Extended Studies (Weeks 4-6)**
- High count rate measurements (pile-up)
- LED pulsing studies (SiPM characterization)
- Temperature-dependent measurements (optional)
- Long-term stability tests

### 5.2 Data Quality Control

**Real-time Monitoring:**
- Count rate stability
- Baseline drift
- Trigger rate
- Dead time fraction

**Quality Metrics:**
- Minimum events per dataset: 10,000
- Maximum pile-up fraction: <5% (for calibration data)
- Baseline stability: RMS < 2 ADC counts
- Trigger efficiency: >95% for events above threshold

**Data Validation Checks:**
- Photopeak positions consistent across runs
- Background subtraction validity
- No saturation in ADC (max counts << 16384)
- Timestamp consistency

### 5.3 Data Organization

**Directory Structure:**
```
data/
├── raw/
│   ├── LYSO/
│   │   ├── Cs137/
│   │   │   ├── run001_waveforms.h5
│   │   │   ├── run001_metadata.json
│   │   │   └── run002_waveforms.h5
│   │   ├── Na22/
│   │   └── ...
│   ├── BGO/
│   ├── NaI/
│   └── Plastic/
├── processed/
│   ├── energy_spectra/
│   ├── pulse_features/
│   └── ml_datasets/
├── calibration/
│   ├── energy_calibration.csv
│   ├── temperature_log.csv
│   └── background_spectra/
└── metadata/
    ├── detector_config.json
    ├── source_info.json
    └── measurement_log.csv
```

**File Formats:**

**HDF5 Structure for Waveforms:**
```python
file.h5
├── /waveforms [N × M array, N=events, M=samples]
├── /timestamps [N array]
├── /baselines [N array]
├── /energies [N array, if computed]
└── /metadata
    ├── scintillator: "LYSO"
    ├── source: "Cs137"
    ├── sampling_rate_MHz: 125
    ├── record_length: 1024
    ├── measurement_date: "2025-10-23"
    ├── temperature_C: 22.5
    └── ...
```

**Metadata JSON:**
```json
{
  "measurement_id": "LYSO_Cs137_001",
  "date": "2025-10-23T14:30:00",
  "scintillator": {
    "type": "LYSO",
    "dimensions_mm": [10, 10, 10],
    "manufacturer": "...",
    "serial": "..."
  },
  "sipm": {
    "type": "...",
    "overvoltage_V": 4.0,
    "temperature_C": 22.5
  },
  "source": {
    "isotope": "Cs137",
    "activity_uCi": 1.0,
    "distance_cm": 10,
    "measurement_duration_s": 3600
  },
  "daq": {
    "model": "DT5825S",
    "sampling_rate_MHz": 125,
    "trigger_threshold_ADC": 50,
    "record_length": 1024,
    "baseline_samples": 100
  },
  "statistics": {
    "total_events": 125463,
    "measurement_time_s": 3600,
    "average_rate_cps": 34.85,
    "dead_time_percent": 1.2
  }
}
```

### 5.4 Data Backup and Version Control

- **Real-time backup** to secondary drive during acquisition
- **Cloud storage** for processed results and code
- **Git repository** for analysis code and documentation
- **Data DOI** upon publication (Zenodo, figshare, or institutional repository)

---

## 6. Analysis Methods

### 6.1 Traditional Spectroscopy Analysis

#### 6.1.1 Energy Calibration

**Method:**
1. Identify photopeaks in each spectrum using peak finding algorithm
2. Fit Gaussian to each photopeak (with background)
3. Create linear calibration: Energy (keV) = m × Channel + b
4. Assess linearity using R² and residuals
5. Repeat for each scintillator

**Expected Photopeaks:**
- Am-241: 59.5 keV
- Co-57: 122, 136 keV
- Cs-137: 662 keV
- Na-22: 511, 1275 keV
- Co-60: 1173, 1332 keV

**Calibration Validation:**
- Chi-square test for linearity
- Cross-validation with independent source
- Uncertainty propagation

**Python Implementation:**
```python
def energy_calibration(spectrum, known_peaks_keV):
    """
    Perform energy calibration from spectrum with known peaks
    Returns calibration coefficients and quality metrics
    """
    # Peak finding
    # Gaussian fitting
    # Linear regression
    # Return: slope, intercept, R², residuals
```

#### 6.1.2 Energy Resolution

**Calculation:**
$$\text{Resolution (\%)} = \frac{\text{FWHM}}{\text{Peak Position}} \times 100$$

**Method:**
1. Fit Gaussian (or Gaussian + low-energy tail) to photopeak
2. Extract FWHM from fit
3. Calculate resolution percentage
4. Plot resolution vs. energy for each scintillator

**Expected Results:**

| Scintillator | Resolution @ 662 keV (typical) |
|--------------|--------------------------------|
| NaI(Tl) | 6-7% |
| LYSO | 8-12% |
| BGO | 10-15% |
| Plastic | >20% (poor) |

**Advanced Fitting:**
- Low-energy tail modeling (incomplete charge collection)
- Background subtraction (Compton continuum)
- Peak overlap deconvolution (Co-60, Na-22)

#### 6.1.3 Detection Efficiency

**Photopeak Efficiency:**
$$\epsilon_{peak} = \frac{\text{Counts in photopeak}}{\text{Total incident photons}}$$

**Relative Efficiency:**
- Compare to reference detector (if available)
- Or compare scintillators relative to each other

**Peak-to-Compton Ratio:**
$$\text{P/C} = \frac{\text{Photopeak counts}}{\text{Compton edge counts}}$$

Higher P/C indicates better energy resolution and full-energy absorption.

#### 6.1.4 Low-Energy Threshold

**Method:**
1. Measure Am-241 (59.5 keV) spectrum
2. Determine detection efficiency vs. energy
3. Define threshold as 50% detection efficiency
4. Compare across scintillators

**Expected:**
- LYSO, NaI: ~20-30 keV threshold
- BGO: ~30-40 keV
- Plastic: ~50-100 keV (poor at low energy)

### 6.2 Pulse Shape Analysis

#### 6.2.1 Feature Extraction

**Timing Features:**

1. **Rise Time (10-90%):**
   $$t_{rise} = t_{90\%} - t_{10\%}$$
   
2. **Fall Time (90-10%):**
   Measure from peak to decay

3. **Decay Constant:**
   Fit exponential to tail:
   $$I(t) = I_0 e^{-t/\tau} + \text{background}$$

4. **Peak Position:**
   Time of maximum amplitude

**Charge Integration Features (for PSD):**

1. **Total Charge:**
   $$Q_{total} = \int_0^T I(t) \, dt$$

2. **Tail Charge:**
   $$Q_{tail} = \int_{t_{gate}}^T I(t) \, dt$$
   where $t_{gate}$ is optimized per scintillator

3. **Tail-to-Total Ratio:**
   $$R_{T/T} = \frac{Q_{tail}}{Q_{total}}$$

**Shape Features:**

1. **Skewness:** Asymmetry of pulse shape
2. **Kurtosis:** "Peakedness" of distribution
3. **Width (FWHM):** Full width at half maximum

**Frequency Domain:**

1. **FFT Power Spectrum:** Identify dominant frequencies
2. **Spectral Centroid:** Weighted average of frequencies

**Feature Vector for ML:**
```python
features = [
    'amplitude',
    'baseline_std',
    'rise_time_10_90',
    'fall_time_90_10',
    'peak_position',
    'total_charge',
    'tail_charge',
    'tail_total_ratio',
    'width_fwhm',
    'skewness',
    'kurtosis',
    'dominant_frequency',
    'decay_constant',
    'rise_slope',
    'fall_slope'
]
# Total: 15 features
```

#### 6.2.2 Traditional Pulse Shape Discrimination (PSD)

**Charge Integration Method:**

Discrimination parameter:
$$\text{PSD} = \frac{Q_{tail}}{Q_{total}}$$

**Application:**
- Plastic scintillator: gamma vs. neutron/beta discrimination
- Expected: Different PSD for Sr-90 beta vs. gamma backgrounds

**Optimization:**
- Vary tail gate start time
- Maximize figure of merit (FOM):
  $$\text{FOM} = \frac{|\mu_1 - \mu_2|}{\text{FWHM}_1 + \text{FWHM}_2}$$

### 6.3 Machine Learning-Based Analysis

#### 6.3.1 Problem Formulation

**Classification Task:**
- **Input:** Raw waveform (1024 samples) OR extracted features (15 values)
- **Output:** Scintillator type (4 classes: LYSO, BGO, NaI, Plastic)
- **Goal:** >95% accuracy with generalization across energies

**Datasets:**
- **Training:** 70% of data, balanced across scintillators and energies
- **Validation:** 15% for hyperparameter tuning
- **Test:** 15% held-out, never seen during training

**Class Balance:**
Ensure equal representation:
- 10,000 pulses × 4 scintillators = 40,000 total
- Stratified sampling across energy ranges

#### 6.3.2 Feature-Based Machine Learning

**Models to Compare:**

1. **Random Forest**
   - Good baseline, interpretable
   - Feature importance analysis
   - Fast training and inference

2. **Gradient Boosting (XGBoost/LightGBM)**
   - Often best performance on tabular data
   - Robust to hyperparameters

3. **Support Vector Machine (SVM)**
   - Good for high-dimensional data
   - Kernel trick for non-linear boundaries

4. **Neural Network (MLP)**
   - Fully connected layers
   - Can learn complex feature interactions

**Hyperparameter Tuning:**
- Grid search or Bayesian optimization
- Cross-validation on training set
- Metrics: Accuracy, F1-score, confusion matrix

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=20)
scores = cross_val_score(rf, X_train, y_train, cv=5)

# XGBoost
xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Compare models...
```

#### 6.3.3 Deep Learning on Raw Waveforms

**1D Convolutional Neural Network (CNN)**

**Architecture:**
```
Input: [batch, 1024, 1] (waveform samples)
├── Conv1D(32 filters, kernel=16, stride=2) + ReLU + MaxPool(2)
├── Conv1D(64 filters, kernel=8, stride=2) + ReLU + MaxPool(2)
├── Conv1D(128 filters, kernel=4, stride=1) + ReLU + MaxPool(2)
├── Flatten
├── Dense(128) + ReLU + Dropout(0.3)
├── Dense(64) + ReLU + Dropout(0.3)
└── Dense(4, activation='softmax')  # 4 scintillator classes
```

**Training:**
- Loss: Categorical cross-entropy
- Optimizer: Adam (lr=0.001)
- Batch size: 64-128
- Epochs: 50-100 with early stopping
- Data augmentation: Time shift, amplitude jitter, noise injection

**Advantages of CNN:**
- Learns features automatically
- No manual feature engineering
- Can capture subtle pulse shape differences
- Potential for higher accuracy

**Implementation Framework:**
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv1D(32, kernel_size=16, strides=2, activation='relu', input_shape=(1024, 1)),
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
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Alternative: Recurrent Neural Networks (LSTM/GRU)**
- Can capture temporal dependencies
- May work well for decay characteristics
- Typically slower than CNN

#### 6.3.4 Model Evaluation

**Metrics:**

1. **Accuracy:** Overall correct classification rate
2. **Precision/Recall/F1 per class:** Identify weak classes
3. **Confusion Matrix:** Where does model fail?
4. **ROC Curves:** One-vs-rest for each class
5. **Inference Time:** For real-time applicability

**Robustness Testing:**

- **Energy dependence:** Performance across different gamma energies
- **Count rate effects:** Does pile-up hurt performance?
- **Noise robustness:** Add synthetic noise, test degradation
- **Cross-source generalization:** Train on some sources, test on others

**Interpretability:**

- **Feature importance:** (Random Forest, XGBoost)
- **Grad-CAM or saliency maps:** Which parts of waveform most important? (CNN)
- **SHAP values:** Global explanation of model

### 6.4 SiPM Characterization: Crosstalk and Afterpulsing

#### 6.4.1 Optical Crosstalk

**Physical Mechanism:**
- Primary avalanche emits photons (hot-carrier luminescence)
- Some photons reach adjacent microcells
- Cause correlated avalanches
- Result: Excess charge beyond primary signal

**Measurement Strategy:**

**Method 1: Single Photon Spectrum Analysis**
- Use low light levels (LED or weak source)
- Measure charge spectrum
- Identify discrete peaks (0, 1, 2, ... photoelectrons)
- Crosstalk causes "shoulder" on 1-p.e. peak

**Analysis:**
```
Crosstalk Probability (P_ct) = Area beyond 1.5 p.e. / Area of 1 p.e. peak
```

**Expected Values:**
- Modern SiPMs: 5-30% crosstalk
- Depends on overvoltage and design

**Method 2: Two-Source Method**
- Measure with weak source (single p.e. regime)
- Measure with strong source (many p.e.)
- Compare expected vs. measured charge
- Excess indicates crosstalk

**Energy-Dependent Crosstalk:**

For scintillation pulses:
1. Bin pulses by energy (ADC amplitude)
2. For each bin, measure effective gain
3. Compare to expected gain
4. Excess gain = crosstalk contribution

**Expected Behavior:**
- Higher crosstalk for high light yield scintillators (NaI, LYSO)
- Crosstalk increases with overvoltage
- May saturate at very high photon counts

**Scintillator Comparison:**
$$\text{Photons/MeV:} \quad \text{NaI(38k)} > \text{LYSO(32k)} >> \text{BGO(8.2k)} > \text{Plastic(10k)}$$

**Hypothesis:** NaI and LYSO will show highest crosstalk effects

#### 6.4.2 Afterpulsing

**Physical Mechanism:**
- Charge carriers trapped in silicon defects
- Released after microseconds to milliseconds
- Cause delayed avalanches
- Appear as fake pulses after real events

**Measurement Strategy:**

**Method 1: Autocorrelation Analysis**
- Measure inter-event time distribution
- Excess at short times (1-100 μs) indicates afterpulsing
- Compare to random (Poisson) expectation

**Implementation:**
```python
def measure_afterpulsing(timestamps, max_delay_us=100):
    """
    Calculate afterpulsing probability from timestamp data
    """
    # Compute time differences between consecutive events
    delta_t = np.diff(timestamps)
    
    # Histogram of inter-event times
    hist, bins = np.histogram(delta_t, bins=np.logspace(-1, 3, 100))
    
    # Fit exponential (random events) + exponential (afterpulsing)
    # P(Δt) = A·exp(-Δt/τ_random) + B·exp(-Δt/τ_afterpulsing)
    
    # Afterpulsing probability = B / (A + B)
    return afterpulsing_probability, tau_afterpulsing
```

**Method 2: Charge Correlation**
- Look for correlation between pulse amplitude and afterpulse rate
- Larger pulses → more trapped carriers → more afterpulsing

**Expected Time Constants:**
- Fast component: 1-10 μs (shallow traps)
- Slow component: 10-100 μs (deep traps)
- Typical afterpulsing: 1-5% per event

**Scintillator-Dependent Effects:**

**Hypothesis:**
- Slow scintillators (BGO, NaI) may mask fast afterpulsing
- Fast scintillators (LYSO, Plastic) more sensitive
- High count rates increase afterpulsing impact

**Temperature Dependence:**
- Afterpulsing decreases at lower temperature
- Trap release rates are thermally activated
- Optional: Measure at multiple temperatures

#### 6.4.3 Saturation and Nonlinearity

**Physical Mechanism:**
- SiPM has finite number of microcells (e.g., 18,000)
- At high photon counts, cells "saturated"
- Response becomes nonlinear

**Saturation Model:**
$$N_{fired} = N_{cells} \cdot (1 - e^{-N_{photons}/N_{cells}})$$

Where:
- $N_{cells}$ = total microcells
- $N_{photons}$ = incident photons
- $N_{fired}$ = fired microcells (measured signal)

**Measurement Strategy:**

**Method 1: Variable Source Distance**
- Measure same gamma line at different distances
- Vary photon flux by ~100×
- Plot SiPM response vs. expected (linear)
- Fit saturation model

**Method 2: Multi-Energy Analysis**
- Use sources spanning 59.5 - 1332 keV
- High energy γ → more photons → potential saturation
- Check if energy calibration remains linear

**Expected Results:**

| Scintillator | Photons @ 662 keV | Saturation Risk |
|--------------|-------------------|-----------------|
| NaI | ~25,000 | **HIGH** |
| LYSO | ~21,000 | **HIGH** |
| BGO | ~5,400 | Low |
| Plastic | ~6,600 | Low |

For 18,000 cell SiPM, expect saturation >10,000 photons

**Correction Algorithm:**
```python
def correct_saturation(measured_signal, n_cells, pde=0.3):
    """
    Apply inverse saturation correction
    """
    # Estimate fired cells
    n_fired = measured_signal / single_cell_gain
    
    # Invert saturation equation
    n_photons = -n_cells * np.log(1 - n_fired/n_cells)
    
    # Account for PDE
    n_photons_corrected = n_photons / pde
    
    return n_photons_corrected
```

**Impact on Energy Resolution:**
- Saturation broadens photopeaks
- Degrades resolution at high energies
- May explain differences from literature values

#### 6.4.4 Dark Count Rate

**Measurement:**
- No radiation source, no light
- Record trigger rate above threshold
- Typical: 10-1000 kHz/mm² (temperature dependent)

**Analysis:**
- Measure vs. threshold
- Characterize noise spectrum
- Check temperature dependence

**Impact:**
- Contributes to baseline fluctuations
- Limits low-energy threshold
- Minimal impact at room temp for keV-range gamma

### 6.5 Pile-up Detection and Correction

#### 6.5.1 Pile-up Physics

**Problem:**
At high count rates, pulses overlap:
- **Type 1:** Second pulse on tail of first
- **Type 2:** Complete overlap of two pulses
- **Result:** Energy distortion, dead time, false counts

**Critical Time Window:**
- Fast scintillator (Plastic, 2.4 ns): ~10 ns
- Intermediate (LYSO, 40 ns): ~200 ns
- Slow (NaI, 230 ns; BGO, 300 ns): ~1-2 μs

**Scintillator-Specific Challenge:**
BGO requires ~10× longer inspection window than LYSO!

#### 6.5.2 Pile-up Detection

**Method 1: Baseline Restoration Check**
- After each pulse, baseline should return to nominal
- If next trigger before baseline restored → pile-up
- Different thresholds per scintillator

**Method 2: Pulse Shape Analysis**
- Fit expected pulse shape (exponential decay)
- Large residuals indicate pile-up
- Use template matching

**Method 3: Second Derivative Test**
- $d²I/dt²$ should be smooth for single pulse
- Sharp features indicate overlapping pulses

**Method 4: Machine Learning Classifier**
- Train binary classifier: clean vs. pile-up
- Features: fit quality, shape parameters, baseline variance
- Can be combined with scintillator classifier

**Implementation:**
```python
def detect_pileup(waveform, baseline, decay_time_ns, threshold=3.0):
    """
    Detect pile-up in waveform
    
    Returns:
        is_pileup: Boolean
        pileup_probability: Float (if using ML)
    """
    # Method 1: Baseline restoration
    tail = waveform[-100:]  # Last 100 samples
    baseline_restored = np.abs(np.mean(tail) - baseline) < threshold
    
    # Method 2: Fit quality
    fitted_pulse = fit_exponential_decay(waveform, decay_time_ns)
    chi_square = np.sum((waveform - fitted_pulse)**2)
    
    # Method 3: Second derivative
    d2_waveform = np.diff(waveform, n=2)
    anomaly_score = np.max(np.abs(d2_waveform))
    
    # Combined criteria
    is_pileup = (not baseline_restored) or (chi_square > threshold_chi2) or (anomaly_score > threshold_d2)
    
    return is_pileup
```

#### 6.5.3 Pile-up Correction

**Approach 1: Rejection (Conservative)**
- Discard all pile-up events
- Pro: Clean data
- Con: Significant dead time at high rates
- Best for: Precise spectroscopy

**Approach 2: Deconvolution (Aggressive)**
- Attempt to separate overlapping pulses
- Use known decay time to model first pulse
- Subtract and analyze remainder

**Deconvolution Algorithm:**
```python
def deconvolve_pileup(waveform, decay_time_ns, sampling_ns=8):
    """
    Attempt to separate two overlapping pulses
    
    Returns:
        pulse1_energy: Energy of first pulse
        pulse2_energy: Energy of second pulse
        success: Boolean (True if deconvolution successful)
    """
    # 1. Identify first pulse peak
    peak1_idx = np.argmax(waveform)
    peak1_amplitude = waveform[peak1_idx]
    
    # 2. Model first pulse (exponential decay from peak)
    decay_samples = int(decay_time_ns / sampling_ns)
    t = np.arange(len(waveform) - peak1_idx) * sampling_ns
    model_pulse1 = peak1_amplitude * np.exp(-t / decay_time_ns)
    
    # 3. Subtract first pulse from waveform
    residual = waveform.copy()
    residual[peak1_idx:] -= model_pulse1
    
    # 4. Check if second pulse exists
    if np.max(residual) > detection_threshold:
        peak2_idx = np.argmax(residual)
        peak2_amplitude = residual[peak2_idx]
        
        # Integrate both pulses for energy
        pulse1_energy = np.sum(model_pulse1)
        pulse2_energy = np.sum(residual[residual > 0])
        
        return pulse1_energy, pulse2_energy, True
    else:
        # False alarm, just one pulse
        return np.sum(waveform), 0, False
```

**Approach 3: Trapezoidal Shaping Optimization**
- DT5825S has digital trapezoidal filter
- Optimize rise/flat/fall times per scintillator
- Trade-off: throughput vs. pile-up rejection

**Scintillator-Specific Parameters:**

| Scintillator | Decay (ns) | Optimal Rise (ns) | Flat (ns) | Dead Time |
|--------------|------------|-------------------|-----------|-----------|
| Plastic | 2.4 | 10-20 | 20-40 | ~100 ns |
| LYSO | 40 | 80-120 | 100-200 | ~500 ns |
| NaI | 230 | 400-600 | 500-1000 | ~2 μs |
| BGO | 300 | 500-800 | 1000-2000 | ~3 μs |

**Approach 4: Maximum Likelihood Estimation**
- Model expected signal + pile-up probability
- Iteratively estimate true event rate
- Statistically correct measured spectrum

#### 6.5.4 Validation and Performance Metrics

**Metrics:**

1. **Dead Time:**
   $$\text{Dead Time (\%)} = \frac{T_{dead}}{T_{total}} \times 100$$

2. **Throughput:**
   $$\text{Measured Rate} = \frac{\text{True Rate}}{1 + \text{True Rate} \times \tau_{dead}}$$

3. **Energy Resolution Degradation:**
   - Compare resolution at low vs. high rates
   - Quantify broadening due to pile-up

4. **Peak Position Shift:**
   - Does pile-up shift photopeak centroids?

**Validation Strategy:**
- Use variable source distances (100× rate range)
- Compare pile-up correction methods
- Benchmark against known activity (decay correction)

**Expected Results:**
- LYSO and Plastic: minimal pile-up up to 50k cps
- BGO and NaI: significant pile-up >10k cps
- Deconvolution improves throughput 20-50%

### 6.6 Comparative Performance Summary

#### 6.6.1 Multi-Dimensional Comparison

Create comprehensive comparison across:

**Dimension 1: Spectroscopic Performance**
- Energy resolution (FWHM %) at 662 keV
- Energy range (low threshold to saturation)
- Peak-to-Compton ratio
- Photopeak efficiency

**Dimension 2: Timing Performance**
- Rise time (10-90%)
- Decay time constant
- Coincidence timing resolution (if measured)
- Pile-up resilience

**Dimension 3: SiPM Compatibility**
- Light yield match to SiPM dynamic range
- Crosstalk impact
- Saturation energy
- Practical operating range

**Dimension 4: Practical Considerations**
- Background/intrinsic activity
- Hygroscopic nature (NaI)
- Cost and availability
- Mechanical robustness

**Dimension 5: Application Suitability**
- Medical imaging (PET, SPECT)
- High-energy physics
- Homeland security
- Environmental monitoring
- Astrophysics

#### 6.6.2 Visualization Strategy

**Radar/Spider Charts:**
Normalize each metric 0-1 and plot on radar chart:
- Energy resolution
- Light yield
- Timing
- Dynamic range
- Cost-effectiveness

**Heat Maps:**
Performance matrix: scintillators × metrics

**Scatter Plots:**
- Energy resolution vs. light yield
- Timing vs. stopping power
- Cost vs. performance

**Decision Trees:**
Guide users to optimal scintillator based on requirements

---

## 7. Expected Results

### 7.1 Energy Calibration and Resolution

**Expected Energy Resolution @ 662 keV:**

| Scintillator | Literature (PMT) | Expected (SiPM) | Factors |
|--------------|------------------|-----------------|---------|
| NaI(Tl) | 6-7% | 7-8% | Slight degradation due to SiPM noise |
| LYSO | 8-10% | 9-12% | Good, but lower light yield than NaI |
| BGO | 10-12% | 12-15% | Low light yield limits resolution |
| Plastic | >20% | >25% | Poor inherent resolution |

**Expected Linearity:**
- All scintillators: R² > 0.999 for calibration curve
- Potential nonlinearity at high energy (NaI, LYSO) due to SiPM saturation

**Expected Low-Energy Threshold:**
- NaI: Best, ~20 keV (limited by intrinsic K-40)
- LYSO: ~25 keV (limited by Lu-176 background)
- BGO: ~30-40 keV
- Plastic: ~50-100 keV

### 7.2 Timing Characteristics

**Expected Decay Time Constants:**

| Scintillator | Literature | Measured (fit) |
|--------------|------------|----------------|
| Plastic | 2.4 ns | 2-3 ns |
| LYSO | 40 ns (fast), 200 ns (slow) | Dual exponential fit |
| NaI | 230 ns | 220-240 ns |
| BGO | 300 ns | 280-320 ns |

**Rise Time Measurements:**
- Plastic: <1 ns (limited by SiPM, not scintillator)
- LYSO: ~1 ns
- NaI, BGO: 2-5 ns

### 7.3 Machine Learning Classification

**Expected Performance:**

**Feature-Based ML:**
- Random Forest: 92-96% accuracy
- XGBoost: 93-97% accuracy
- Training time: seconds to minutes
- Inference: <1 ms per pulse

**CNN on Raw Waveforms:**
- Accuracy: 95-99% (higher than feature-based)
- Training time: hours (GPU required)
- Inference: 1-10 ms per pulse (depending on implementation)

**Confusion Matrix Prediction:**
Most errors expected between:
- LYSO ↔ Plastic (both fast decay)
- NaI ↔ BGO (both slow, similar decay times)

**Energy Robustness:**
- High accuracy across 122-1332 keV
- Potential degradation at very low energy (<100 keV)

**Real-World Application:**
- Can identify unknown scintillator from ~10 pulses
- Useful for detector verification in field

### 7.4 SiPM Characterization

**Expected Crosstalk:**
- Typical: 10-25%
- Higher for NaI and LYSO (high photon counts)
- Increases with overvoltage

**Expected Afterpulsing:**
- Probability: 1-5% per event
- Time constant: 10-100 μs
- Larger effect with high amplitude pulses

**Expected Saturation:**
- NaI: Saturation begins ~800 keV
- LYSO: Saturation begins ~1000 keV
- BGO, Plastic: Minimal saturation up to 1.3 MeV

**Impact on Resolution:**
- Crosstalk: ~5-10% degradation
- Afterpulsing: Minimal (low rate)
- Saturation: Up to 20% degradation at high E

### 7.5 Pile-up Correction

**Expected Improvement:**

| Scintillator | Count Rate (cps) | Pile-up % (raw) | After Correction |
|--------------|------------------|-----------------|------------------|
| Plastic | 50,000 | 2% | <0.5% |
| LYSO | 20,000 | 5% | 1-2% |
| NaI | 10,000 | 10% | 3-5% |
| BGO | 5,000 | 15% | 5-8% |

**Throughput Improvement:**
- Deconvolution method: 20-50% more usable events
- Peak position restoration: <5% error
- Energy resolution preservation: >90%

### 7.6 Comparative Ranking

**Overall Performance (1=Best, 4=Worst):**

| Application | 1st | 2nd | 3rd | 4th |
|-------------|-----|-----|-----|-----|
| **Energy Resolution** | NaI | LYSO | BGO | Plastic |
| **Timing/Speed** | Plastic | LYSO | NaI | BGO |
| **Light Yield** | NaI | LYSO | Plastic | BGO |
| **High-Z (stopping power)** | BGO | LYSO | NaI | Plastic |
| **Low Background** | Plastic | BGO | NaI | LYSO |
| **SiPM Compatibility** | BGO | Plastic | LYSO | NaI |
| **Cost** | Plastic | NaI | BGO | LYSO |

**Application-Specific Recommendations:**
- **PET Imaging:** LYSO (timing + sensitivity)
- **Gamma Spectroscopy:** NaI (resolution)
- **High Energy (MeV):** BGO (stopping power)
- **Particle Detection:** Plastic (PSD, speed)
- **Portable/Field:** Plastic or BGO (no hygroscopy, SiPM works well)

---

## 8. Publication Strategy

### 8.1 Target Journals and Impact

**Primary Target Journals:**

**Tier 1 (High Impact):**
1. **IEEE Transactions on Nuclear Science**
   - Impact Factor: ~1.8
   - Focus: Instrumentation, detector development
   - Typical acceptance: 30-40%

2. **Nuclear Instruments and Methods in Physics Research A**
   - Impact Factor: ~1.5
   - Focus: Detector physics, methodology
   - Very appropriate for this work

3. **Journal of Instrumentation (JINST)**
   - Impact Factor: ~1.3
   - Focus: Experimental methods
   - Open access option

**Tier 2 (Specialized):**
4. **Physics in Medicine & Biology**
   - If emphasizing medical imaging applications

5. **Sensors**
   - Open access, broader audience
   - Good for ML/application focus

### 8.2 Proposed Publications

**Paper 1: Comprehensive Comparison Study** (Main paper)

**Title:** *"Comprehensive Characterization of LYSO, BGO, NaI(Tl), and Plastic Scintillators Coupled to Silicon Photomultipliers: A Digital Pulse Processing Approach"*

**Target:** *Nuclear Instruments and Methods A* or *IEEE Trans. Nucl. Sci.*

**Structure:**
1. Introduction
   - SiPM advantages and challenges
   - Need for systematic comparison
   - Previous work limitations

2. Experimental Setup
   - Detector configurations
   - DAQ system
   - Calibration sources

3. Energy Calibration and Resolution
   - Calibration curves
   - Resolution vs. energy
   - Comparative performance

4. Pulse Shape Characteristics
   - Timing measurements
   - Decay constants
   - Feature extraction

5. SiPM Nonlinear Effects
   - Saturation measurements
   - Impact on resolution
   - Crosstalk characterization

6. Digital Pulse Processing Optimization
   - Filter parameter optimization
   - Pile-up handling
   - Dead time analysis

7. Comparative Discussion
   - Performance matrix
   - Application recommendations

8. Conclusions

**Estimated Length:** 12-15 pages
**Estimated Timeline:** 3-4 months from data completion
**Expected Impact:** High citations (benchmark dataset)

---

**Paper 2: Machine Learning Classification** (Novel methodology)

**Title:** *"Machine Learning-Based Scintillator Identification from Raw SiPM Pulse Waveforms"*

**Target:** *IEEE Transactions on Nuclear Science* or *Journal of Instrumentation*

**Structure:**
1. Introduction
   - ML in nuclear instrumentation
   - Problem statement
   - Potential applications

2. Dataset and Features
   - Waveform database
   - Feature extraction methods
   - Data preprocessing

3. Machine Learning Models
   - Feature-based classifiers
   - Deep learning (CNN)
   - Training methodology

4. Results
   - Classification accuracy
   - Confusion matrices
   - Energy robustness
   - Inference speed

5. Interpretation and Analysis
   - Feature importance
   - Learned representations
   - Physical insights

6. Real-World Application Demonstration
   - Blind test
   - Field deployment considerations

7. Conclusions and Future Work

**Estimated Length:** 8-10 pages
**Estimated Timeline:** 2-3 months after Paper 1 submitted
**Expected Impact:** High (novel application of ML, reusable methods)

---

**Paper 3: Pile-up Correction** (Technical focus)

**Title:** *"Scintillator-Specific Pile-up Correction Algorithms for High Count Rate SiPM-Based Gamma Spectroscopy"*

**Target:** *Nuclear Instruments and Methods A*

**Structure:**
1. Introduction
   - Pile-up problem
   - Scintillator decay time differences
   - Importance for high-rate applications

2. Pile-up Detection Methods
   - Algorithm descriptions
   - Performance metrics

3. Correction Strategies
   - Deconvolution approach
   - ML-based correction
   - Trapezoidal shaping optimization

4. Experimental Validation
   - Variable count rate tests
   - Energy resolution preservation
   - Throughput improvement

5. Scintillator-Specific Results
   - LYSO vs. BGO comparison
   - Practical recommendations

6. Conclusions

**Estimated Length:** 8-10 pages
**Estimated Timeline:** Can be parallel with Paper 2
**Expected Impact:** Moderate (practical value for high-rate applications)

---

**Optional Paper 4: Conference Proceedings**

**Title:** *"Open-Source Analysis Toolkit for SiPM-Based Scintillation Detectors"*

**Target:** IEEE NSS/MIC Conference, ANIMMA, or similar

**Focus:**
- Software release announcement
- Tutorial-style description
- Community engagement

**Timeline:** Alongside Paper 1 or 2 submission

### 8.3 Authorship and Collaboration

**Lead Author:** You (primary investigator, data collection, analysis)

**Potential Co-authors:**
- Supervisor/PI
- SiPM manufacturer collaborator (optional, for technical specs)
- ML expert (if significant collaboration on Paper 2)

**Author Contribution Statement:**
- Design, data collection, analysis, writing: Lead author
- Supervision, funding, conceptual guidance: PI
- Specific technical contributions: As appropriate

### 8.4 Open Science Approach

**Data Sharing:**
- Deposit processed datasets on Zenodo or similar repository
- Assign DOI upon publication
- Include:
  - Calibrated energy spectra
  - Representative waveforms (anonymized subset)
  - Extracted feature vectors for ML

**Code Sharing:**
- GitHub repository with full analysis code
- Jupyter notebooks for reproducibility
- Requirements.txt and Docker container for environment

**Benefits:**
- Increases citations and impact
- Enables validation by others
- Supports community advancement
- Required by many funding agencies

### 8.5 Timeline and Milestones

**Month 1-2: Data Collection**
- Weeks 1-2: Initial survey, optimization
- Weeks 3-6: Systematic data collection
- Weeks 7-8: Extended studies (pile-up, LED)

**Month 3-4: Analysis**
- Weeks 9-10: Traditional analysis (calibration, resolution)
- Weeks 11-12: Pulse shape analysis, SiPM characterization
- Weeks 13-14: ML model development and training
- Weeks 15-16: Pile-up correction implementation

**Month 5: Paper 1 Writing**
- Weeks 17-18: Draft manuscript
- Weeks 19-20: Internal review, revisions

**Month 6: Submission and Paper 2**
- Week 21: Submit Paper 1
- Weeks 21-24: Draft Paper 2 (ML classification)

**Month 7-8: Reviews and Revisions**
- Respond to Paper 1 reviews
- Submit Paper 2
- Begin Paper 3 if desired

**Month 9-12: Acceptance and Beyond**
- Paper acceptance and publication
- Conference presentations
- Software release and community engagement

---

## 9. Software Implementation Plan

### 9.1 Repository Structure

```
sipm-scintillator-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml (conda)
├── Dockerfile
├── setup.py
│
├── docs/
│   ├── installation.md
│   ├── quickstart.md
│   ├── tutorials/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_energy_calibration.ipynb
│   │   ├── 03_pulse_shape_analysis.ipynb
│   │   ├── 04_ml_classification.ipynb
│   │   ├── 05_pileup_correction.ipynb
│   │   └── 06_sipm_characterization.ipynb
│   └── api_reference/
│
├── data/
│   ├── raw/  (not in git, .gitignore)
│   ├── processed/
│   ├── example_datasets/  (small samples for testing)
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── waveform_loader.py
│   │   └── data_formats.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── energy_calibration.py
│   │   └── peak_finding.py
│   ├── pulse_analysis/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── pulse_fitting.py
│   │   └── psd_traditional.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── sipm/
│   │   ├── __init__.py
│   │   ├── crosstalk.py
│   │   ├── afterpulsing.py
│   │   └── saturation.py
│   ├── pileup/
│   │   ├── __init__.py
│   │   ├── detection.py
│   │   └── correction.py
│   └── visualization/
│       ├── __init__.py
│       ├── spectra.py
│       └── comparative_plots.py
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_comprehensive_characterization.ipynb
│   ├── 03_ml_development.ipynb
│   ├── 04_pileup_studies.ipynb
│   ├── 05_paper_figures.ipynb
│   └── 06_supplementary_analysis.ipynb
│
├── scripts/
│   ├── batch_process_data.py
│   ├── train_ml_models.py
│   ├── generate_report.py
│   └── optimize_filters.py
│
├── tests/
│   ├── test_calibration.py
│   ├── test_pulse_analysis.py
│   ├── test_ml_models.py
│   └── test_pileup.py
│
└── results/
    ├── figures/
    ├── tables/
    └── models/  (trained ML models)
```

### 9.2 Core Modules to Implement

**Module 1: Data I/O and Preprocessing**
```python
# src/io/waveform_loader.py

class WaveformLoader:
    """Load and preprocess waveform data from various formats"""
    - load_h5()
    - load_npy()
    - load_csv()
    - batch_loader()
    - apply_baseline_correction()
    - apply_noise_filter()
```

**Module 2: Energy Calibration**
```python
# src/calibration/energy_calibration.py

class EnergyCalibrator:
    """Perform energy calibration"""
    - find_photopeaks()
    - fit_gaussian()
    - calibrate_spectrum()
    - calculate_resolution()
    - plot_calibration_curve()
```

**Module 3: Pulse Shape Analysis**
```python
# src/pulse_analysis/feature_extraction.py

class PulseFeatureExtractor:
    """Extract comprehensive pulse features"""
    - extract_timing_features()
    - extract_charge_integration_features()
    - extract_shape_features()
    - extract_frequency_features()
    - get_feature_vector()
```

**Module 4: Machine Learning**
```python
# src/ml/models.py

class ScintillatorClassifier:
    """ML-based scintillator classification"""
    - build_feature_model()  # Random Forest, XGBoost
    - build_cnn_model()      # Deep learning
    - train()
    - predict()
    - evaluate()
    - explain_predictions()
```

**Module 5: SiPM Characterization**
```python
# src/sipm/crosstalk.py

class CrosstalkAnalyzer:
    """Analyze optical crosstalk"""
    - measure_single_pe_spectrum()
    - calculate_crosstalk_probability()
    - energy_dependent_crosstalk()
    - plot_crosstalk_vs_amplitude()
```

**Module 6: Pile-up Handling**
```python
# src/pileup/correction.py

class PileupCorrector:
    """Detect and correct pile-up events"""
    - detect_pileup()
    - deconvolve_pulses()
    - correct_spectrum()
    - calculate_dead_time()
```

### 9.3 Jupyter Notebook Plan

**Notebook 1: Data Loading and Exploration**
- Load sample waveforms
- Visualize pulse shapes
- Basic statistics
- Quality checks

**Notebook 2: Energy Calibration**
- Peak finding for all sources
- Calibration curve generation
- Resolution calculations
- Comparative plots

**Notebook 3: Pulse Shape Analysis**
- Feature extraction demonstration
- Timing measurements
- Decay curve fitting
- Traditional PSD

**Notebook 4: Machine Learning Classification**
- Dataset preparation
- Model training (feature-based and CNN)
- Evaluation and confusion matrices
- Robustness testing

**Notebook 5: Pile-up Studies**
- Pile-up detection methods
- Correction algorithm comparison
- Rate-dependent analysis
- Dead time calculations

**Notebook 6: SiPM Characterization**
- Crosstalk analysis
- Afterpulsing measurements
- Saturation studies
- Nonlinearity correction

**Notebook 7: Comprehensive Comparison**
- Multi-dimensional performance comparison
- Radar charts and heat maps
- Application recommendations
- Final summary

**Notebook 8: Paper Figures**
- Generate all publication-quality figures
- Consistent styling
- High-resolution exports

### 9.4 Key Dependencies

```
# requirements.txt

# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
h5py>=3.1.0

# Plotting
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Machine learning
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Deep learning (optional, for CNN)
tensorflow>=2.8.0
# OR
pytorch>=1.10.0

# Signal processing
PyWavelets>=1.1.1
peakutils>=1.3.0

# Optimization
hyperopt>=0.2.5  # for hyperparameter tuning

# Interpretability
shap>=0.40.0     # for model explanation

# Documentation
jupyter>=1.0.0
nbconvert>=6.0.0
sphinx>=4.0.0    # for docs

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0

# Utilities
tqdm>=4.62.0     # progress bars
pyyaml>=5.4.0    # config files
```

### 9.5 Testing Strategy

**Unit Tests:**
- Test each function independently
- Mock data for reproducibility
- Edge cases and error handling

**Integration Tests:**
- End-to-end pipeline tests
- Real data subset
- Performance benchmarks

**Continuous Integration:**
- GitHub Actions for automated testing
- Run on multiple Python versions (3.8-3.11)
- Code coverage reporting

**Example Test:**
```python
# tests/test_calibration.py

import pytest
import numpy as np
from src.calibration import EnergyCalibrator

def test_energy_calibration():
    """Test energy calibration with known peaks"""
    # Mock spectrum with peaks at known channels
    spectrum = create_mock_spectrum(peaks=[100, 200, 300])
    known_energies = [122, 511, 662]  # keV
    
    calibrator = EnergyCalibrator()
    slope, intercept = calibrator.calibrate(spectrum, known_energies)
    
    # Check calibration accuracy
    assert slope == pytest.approx(expected_slope, rel=0.01)
    assert intercept == pytest.approx(expected_intercept, rel=0.01)
```

### 9.6 Documentation Plan

**README.md:**
- Project overview
- Installation instructions
- Quick start guide
- Citation information

**Tutorials:**
- Step-by-step guides
- Runnable Jupyter notebooks
- Common use cases

**API Reference:**
- Auto-generated from docstrings (Sphinx)
- Class and function documentation
- Examples for each major function

**Contributing Guide:**
- How to contribute
- Code style (PEP 8)
- Pull request process

**Zenodo/Figshare:**
- Data repository with DOI
- Link from GitHub README
- Version control for data releases

---

## 10. Timeline & Milestones

### 10.1 Detailed Gantt Chart

```
Month 1-2: DATA COLLECTION PHASE
├── Week 1-2: Setup and Initial Survey
│   ├── [ ] Verify all detectors and sources
│   ├── [ ] Optimize trigger thresholds
│   ├── [ ] Test data acquisition chain
│   └── [ ] Record initial spectra for all combinations
│
├── Week 3-4: Systematic Calibration Data
│   ├── [ ] LYSO: All 6 sources (Cs137, Na22, Co60, Co57, Am241, Sr90)
│   ├── [ ] BGO: All 6 sources
│   ├── [ ] NaI: All 6 sources
│   ├── [ ] Plastic: All 6 sources
│   └── [ ] Background measurements for each
│
├── Week 5-6: Waveform Collection for ML
│   ├── [ ] Collect 10k+ pulses per scintillator
│   ├── [ ] Ensure energy stratification
│   ├── [ ] Quality control checks
│   └── [ ] Organize data repository
│
└── Week 7-8: Extended Studies
    ├── [ ] High count rate measurements (pile-up)
    ├── [ ] LED pulsing studies (optional)
    ├── [ ] Temperature sweeps (if equipment available)
    └── [ ] Data validation and backup

Month 3-4: ANALYSIS PHASE
├── Week 9-10: Traditional Analysis
│   ├── [ ] Energy calibration for all scintillators
│   ├── [ ] Resolution calculations
│   ├── [ ] Efficiency and P/C ratios
│   └── [ ] Generate spectroscopy figures
│
├── Week 11-12: Pulse Shape Analysis
│   ├── [ ] Feature extraction from waveforms
│   ├── [ ] Timing measurements (rise/fall times)
│   ├── [ ] Decay curve fitting
│   └── [ ] Traditional PSD implementation
│
├── Week 13-14: ML Development
│   ├── [ ] Dataset preparation and splitting
│   ├── [ ] Feature-based models (RF, XGBoost)
│   ├── [ ] CNN architecture and training
│   ├── [ ] Model evaluation and optimization
│   └── [ ] Robustness testing
│
└── Week 15-16: SiPM and Pile-up Studies
    ├── [ ] Crosstalk characterization
    ├── [ ] Afterpulsing analysis
    ├── [ ] Saturation measurements
    ├── [ ] Pile-up detection and correction
    └── [ ] Comparative performance summary

Month 5: PAPER 1 WRITING
├── Week 17-18: Draft Manuscript
│   ├── [ ] Introduction and background
│   ├── [ ] Experimental setup section
│   ├── [ ] Results: Calibration and resolution
│   ├── [ ] Results: Pulse shape characteristics
│   ├── [ ] Results: SiPM effects
│   └── [ ] Discussion and conclusions
│
└── Week 19-20: Internal Review
    ├── [ ] Co-author feedback
    ├── [ ] Figure polishing
    ├── [ ] Supplementary materials
    └── [ ] Final revisions

Month 6: SUBMISSION AND PAPER 2
├── Week 21: Paper 1 Submission
│   ├── [ ] Format for target journal
│   ├── [ ] Cover letter
│   ├── [ ] Submit to journal
│   └── [ ] Update preprint (arXiv)
│
└── Week 22-24: Paper 2 Drafting (ML Classification)
    ├── [ ] Introduction and motivation
    ├── [ ] Methods: Dataset and models
    ├── [ ] Results: Classification performance
    ├── [ ] Discussion: Applications and insights
    └── [ ] Draft for internal review

Month 7-8: REVIEWS AND PAPER 3
├── Week 25-28: Paper 1 Reviews
│   ├── [ ] Respond to reviewer comments
│   ├── [ ] Additional analysis if needed
│   ├── [ ] Revise manuscript
│   └── [ ] Resubmit
│
└── Parallel: Paper 2 Submission + Paper 3 Draft
    ├── [ ] Submit Paper 2 (ML classification)
    ├── [ ] Begin Paper 3 (Pile-up correction)
    └── [ ] Conference abstract submission

Month 9-12: ACCEPTANCE AND OUTREACH
├── [ ] Paper 1 acceptance and publication
├── [ ] Paper 2 reviews and revisions
├── [ ] Paper 3 completion
├── [ ] GitHub repository public release
├── [ ] Data deposit on Zenodo
├── [ ] Conference presentation(s)
├── [ ] Blog post / press release
└── [ ] Community engagement

```

### 10.2 Critical Path

**Must-Have for Minimal Viable Publication:**
1. ✅ Complete energy calibration (4 scintillators × 6 sources)
2. ✅ Energy resolution measurements
3. ✅ Pulse shape characterization
4. ✅ ML classification >90% accuracy
5. ✅ Basic crosstalk quantification
6. ✅ Pile-up correction demonstration

**Nice-to-Have Enhancements:**
- Temperature-dependent measurements
- Coincidence timing with Na-22
- LED pulsing studies
- High-rate performance (>100k cps)
- Field deployment demonstration

### 10.3 Risk Management

**Risk 1: Equipment Failure**
- Mitigation: Regular backups, redundant storage
- Contingency: Repeat measurements if needed (sources stable)

**Risk 2: Data Quality Issues**
- Mitigation: Real-time monitoring, quality checks
- Contingency: Adjust acquisition parameters, extended measurement time

**Risk 3: ML Models Underperform**
- Mitigation: Multiple model architectures, hyperparameter tuning
- Contingency: Focus on traditional methods, simpler ML (still publishable)

**Risk 4: Reviewer Requests Major Revisions**
- Mitigation: Internal review before submission
- Contingency: Budget extra time (2-4 months) for major revisions

**Risk 5: Scope Creep**
- Mitigation: Stick to core objectives, defer secondary analysis
- Contingency: Additional papers for extended work

### 10.4 Success Metrics

**Quantitative:**
- [ ] 3 peer-reviewed publications accepted
- [ ] >100 citations within 2 years
- [ ] GitHub repository: >50 stars, >20 forks
- [ ] Data DOI: >100 downloads

**Qualitative:**
- [ ] Positive reviewer feedback on novelty
- [ ] Industry or academic collaborations initiated
- [ ] Invited talks at conferences
- [ ] Adoption of methods by other research groups

---

## 11. Computational Resources

### 11.1 Hardware Requirements

**For Data Acquisition:**
- PC with CAEN DT5825S support
- Linux recommended (better driver support)
- Storage: 500 GB - 2 TB (depending on waveform count)

**For Analysis:**
- CPU: Modern multi-core (8+ cores recommended)
- RAM: 16-32 GB (for ML training)
- GPU: Optional but recommended for CNN training (NVIDIA with CUDA)
- Storage: SSD for fast I/O

**For ML Training:**
- GPU highly recommended: NVIDIA RTX 3060 or better
- Cloud option: Google Colab Pro, AWS, or university cluster

### 11.2 Software Environment

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- macOS (compatible, but GPU support limited)
- Windows (WSL2 for best compatibility)

**Python Version:**
- Python 3.8 - 3.11 (3.10 recommended)

**Key Libraries:**
See Section 9.4 for complete list

**Development Tools:**
- Jupyter Lab or Jupyter Notebook
- VS Code or PyCharm
- Git for version control
- Docker for reproducible environment

### 11.3 Computational Time Estimates

**Data Processing:**
- Waveform preprocessing: ~1 hour per 100k pulses
- Feature extraction: ~2-4 hours for full dataset
- Energy calibration: ~30 min per scintillator

**Machine Learning:**
- Feature-based training: Minutes (CPU sufficient)
- CNN training: 1-4 hours (GPU), 12-24 hours (CPU)
- Hyperparameter optimization: 4-12 hours

**Analysis and Visualization:**
- Spectroscopy plots: <1 hour
- Comprehensive comparison: 2-4 hours
- Paper figures: 1-2 days

**Total Estimated Computation Time:** 1-2 weeks of CPU time

---

## 12. Budget and Resources (If Applicable)

### 12.1 Equipment Costs

**Assumed Existing:**
- SiPM and readout electronics: $0
- Scintillators: $0
- CAEN DT5825S: $0
- Radiation sources: $0 (already available)

**Potential Additional Costs:**
- LED pulser for SiPM studies: ~$200-500
- Temperature-controlled enclosure: ~$500-2000 (optional)
- High-activity sources (if needed): ~$500-1000

### 12.2 Software Costs

**Free/Open Source:**
- Python and all scientific libraries: $0
- Jupyter, Git, etc.: $0
- Basic compute: $0 (use existing workstation)

**Optional Paid:**
- GitHub Copilot: ~$10/month (coding assistant)
- Cloud GPU (if needed): ~$1-2/hour, ~$50-200 total
- Plotly Pro (advanced plotting): ~$400/year (optional)

### 12.3 Publication Costs

**Journal Fees:**
- Traditional journals (IEEE, NIM): ~$0-500 per paper
- Open access: ~$1000-3000 per paper (optional)
- Preprints (arXiv): $0

**Estimated Total:** $0-3000 depending on open access choice

### 12.4 Conference Costs (Optional)

**IEEE NSS/MIC or similar:**
- Registration: ~$500-800
- Travel: $500-2000 (depending on location)
- Accommodation: $500-1000

**Total Conference:** ~$1500-3800 (if attending)

---

## 13. Ethical and Safety Considerations

### 13.1 Radiation Safety

**Compliance:**
- [ ] All sources licensed and properly stored
- [ ] Radiation safety training completed
- [ ] Dosimetry if required
- [ ] Work area properly posted

**Best Practices:**
- ALARA (As Low As Reasonably Achievable)
- Minimize exposure time
- Maximize distance from sources
- Use shielding when appropriate

### 13.2 Data Management

**Privacy:**
- No personal data collected
- Metadata: Equipment settings, environmental conditions only

**Data Integrity:**
- Raw data preserved
- Analysis pipeline documented
- Version control for code

**Long-term Storage:**
- Institutional repository
- Public data archive (Zenodo)
- Minimum 10-year retention

### 13.3 Research Ethics

**Conflicts of Interest:**
- Declare any industry relationships
- Funding sources acknowledged

**Authorship:**
- Follow ICMJE guidelines
- Contributions clearly stated
- No ghost or guest authorship

**Data Sharing:**
- Commit to open science
- Make data and code publicly available upon publication
- Encourage reproducibility

---

## 14. Conclusion and Next Steps

### 14.1 Study Impact Summary

This comprehensive study will:

1. **Advance Scientific Knowledge:**
   - First systematic comparison of these 4 scintillators with SiPMs
   - Quantify SiPM-specific effects across different light yields
   - Demonstrate ML applicability to scintillation detection

2. **Provide Practical Value:**
   - Guide detector selection for various applications
   - Offer optimized digital processing parameters
   - Enable real-time scintillator identification

3. **Benefit the Community:**
   - Open-source analysis toolkit
   - Public benchmark dataset
   - Reproducible methodology

4. **Generate Publications:**
   - 2-3 high-quality peer-reviewed papers
   - Conference presentations
   - Strong citation potential

### 14.2 Immediate Next Steps

**Week 1:**
- [ ] Finalize experimental setup and verify all equipment
- [ ] Organize data storage structure
- [ ] Begin initial survey measurements
- [ ] Set up version control (Git repository)

**Week 2:**
- [ ] Start systematic data collection
- [ ] Implement basic data quality monitoring
- [ ] Begin preliminary analysis code development

**Week 3-4:**
- [ ] Continue data collection
- [ ] Parallel: Develop analysis framework
- [ ] Test feature extraction on initial data

**Month 2:**
- [ ] Complete data collection
- [ ] Begin main analysis phase
- [ ] Regular progress meetings/check-ins

### 14.3 Success Indicators (3-Month Check)

✅ **On Track If:**
- Data collection >75% complete
- Energy calibration working for all scintillators
- ML classification preliminary results >85% accuracy
- Clear path to Paper 1

⚠️ **Warning Signs:**
- Major equipment issues causing delays
- Data quality problems
- ML models not converging
- Scope expanding uncontrollably

### 14.4 Long-Term Vision

**Year 1:** Complete study, publish Papers 1-2, present at conference
**Year 2:** Paper 3, software maturity, community adoption
**Year 3:** Follow-up studies, collaborations, applied projects

**Potential Extensions:**
- More scintillator types (CsI, GAGG, etc.)
- Multiple SiPM models comparison
- Real-world application deployment (medical imaging, security)
- ML transfer learning to new detector systems

### 14.5 Contact and Collaboration

**GitHub:** [Repository will be created]
**Email:** [Your contact]
**Lab/Institution:** [Your affiliation]

**Collaboration Opportunities:**
- Data sharing and cross-validation
- ML model improvements
- Application-specific optimization
- Multi-site validation studies

---

## 15. Appendices

### Appendix A: Scintillator Property Tables

**(Detailed tables with all known properties)**

### Appendix B: Source Decay Schemes

**(Energy level diagrams and branching ratios)**

### Appendix C: Code Snippets

**(Key algorithms in pseudocode)**

### Appendix D: Glossary

**ADC:** Analog-to-Digital Converter  
**Compton Edge:** Maximum energy transferred in Compton scattering  
**Crosstalk:** Avalanche in one SiPM cell triggering adjacent cells  
**FWHM:** Full Width at Half Maximum  
**Photopeak:** Full-energy absorption peak in gamma spectrum  
**PSD:** Pulse Shape Discrimination  
**SiPM:** Silicon Photomultiplier  

### Appendix E: References

**(Key papers and resources - to be expanded)**

1. Advantages of SiPMs over PMTs
2. Scintillator properties compendium
3. Machine learning in nuclear instrumentation
4. Pile-up correction methods
5. Digital pulse processing techniques

---

## Document Control

**Version:** 1.0  
**Date:** October 23, 2025  
**Author:** [Your name]  
**Status:** Draft for Implementation  

**Revision History:**
- v1.0 (2025-10-23): Initial comprehensive protocol

**Approval:**
- [ ] Primary Investigator
- [ ] Safety Officer (if required)
- [ ] Collaborators

---

**END OF DOCUMENT**

*This study protocol is a living document and will be updated as the research progresses.*