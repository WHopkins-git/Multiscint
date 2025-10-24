# Claude Code Prompt: SiPM Detector Analysis Framework (CORRECTED)

## 🎯 Critical Alignment

This prompt is now **fully aligned** with the research protocol in `SiPM_Detector_Study_Complete_Protocol.md`. 

**NO SCOPE CREEP. NO NEW RESEARCH DIRECTIONS.**

This prompt implements exactly what was specified in the protocol and notebook specifications - nothing more, nothing less.

---

## 📚 Context Documents

You have three specification documents that define a complete, focused research project:
1. **SiPM_Detector_Study_Complete_Protocol.md** - Comprehensive research methodology (2-3 papers)
2. **Jupyter_Notebook_Specifications.md** - Detailed implementation specs for 8 notebooks
3. **README.md** - Project overview

## 🎯 Your Task: Implement the Specified Research Plan

Generate a complete, production-ready Python codebase that implements **exactly** what is defined in these specifications.

### Machine Learning Scope (AS SPECIFIED IN PROTOCOL SECTION 6.3)

Implement these models **only**:

#### Feature-Based ML (6.3.2):
- ✅ Random Forest
- ✅ Gradient Boosting (XGBoost)
- ✅ Support Vector Machine (SVM)
- ✅ Neural Network (MLP)

#### Deep Learning on Raw Waveforms (6.3.3):
- ✅ 1D Convolutional Neural Network (CNN) - **specific architecture from spec**

```python
# From Protocol Section 6.3.3 - Use this EXACT architecture
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
```

**THAT'S IT. No more ML models.**

The goal is 92-99% accuracy with these models. Additional models are **future work**, not this project.

---

## 📁 Project Structure (From Specifications)

```
sipm-analysis/
├── src/
│   ├── io/
│   │   ├── waveform_loader.py          # Load HDF5, NPY, CSV waveforms
│   │   └── data_formats.py             # Data structures
│   ├── calibration/
│   │   ├── energy_calibration.py       # Linear calibration, resolution
│   │   └── peak_finding.py             # Automated photopeak identification
│   ├── pulse_analysis/
│   │   ├── feature_extraction.py       # 15 features from protocol
│   │   ├── pulse_fitting.py            # Exponential decay fitting
│   │   └── psd_traditional.py          # Charge integration PSD
│   ├── ml/
│   │   ├── models.py                   # Base classes
│   │   ├── feature_based.py            # RF, XGBoost, SVM, MLP
│   │   ├── cnn_model.py                # The CNN from protocol
│   │   ├── training.py                 # Training loops
│   │   └── evaluation.py               # Metrics, confusion matrices
│   ├── sipm/
│   │   ├── crosstalk.py                # Optical crosstalk measurement
│   │   ├── afterpulsing.py             # Afterpulsing analysis
│   │   └── saturation.py               # Nonlinearity modeling
│   ├── pileup/
│   │   ├── detection.py                # Pile-up detection methods
│   │   └── correction.py               # Deconvolution algorithms
│   └── visualization/
│       ├── spectra.py                  # Energy spectrum plotting
│       └── comparative_plots.py        # Multi-scintillator comparisons
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_energy_calibration.ipynb
│   ├── 03_pulse_shape_analysis.ipynb
│   ├── 04_ml_classification.ipynb      # All 5 models from protocol
│   ├── 05_pileup_correction.ipynb
│   ├── 06_sipm_characterization.ipynb
│   ├── 07_comprehensive_comparison.ipynb
│   └── 08_paper_figures.ipynb
├── tests/
├── configs/
│   └── model_config.yaml               # Hyperparameters for the 5 models
└── scripts/
    ├── batch_process_data.py
    ├── train_ml_models.py
    └── generate_report.py
```

**Note:** This is the structure from the specifications. No additional ML model files.

---

## 🔬 Implementation Requirements

### 1. Data Loading (src/io/)

Implement `WaveformLoader` class as specified in Notebook 1:
- Load HDF5, NPY, CSV formats
- Calculate baselines from first 50 samples
- Support batch loading
- Metadata extraction

### 2. Energy Calibration (src/calibration/)

From Notebook 2 specifications:
- Automated peak finding (scipy.signal.find_peaks)
- Gaussian fitting with background
- Linear calibration: Energy = m × Channel + b
- Resolution calculation: FWHM/peak × 100%

### 3. Pulse Shape Analysis (src/pulse_analysis/)

From Notebook 3 specifications, extract **these 15 features**:
1. amplitude
2. baseline_std
3. rise_time_10_90
4. fall_time_90_10
5. peak_position
6. total_charge
7. tail_charge
8. tail_total_ratio
9. width_fwhm
10. skewness
11. kurtosis
12. dominant_frequency
13. decay_constant (from exponential fit)
14. rise_slope
15. fall_slope

### 4. Machine Learning (src/ml/)

#### Feature-Based Models

```python
# src/ml/feature_based.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

class FeatureBasedClassifiers:
    """
    Implement the 4 feature-based models from Protocol Section 6.3.2
    """
    
    def train_random_forest(self, X_train, y_train):
        """Random Forest with hyperparameter tuning"""
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        rf.fit(X_train, y_train)
        return rf
    
    def train_xgboost(self, X_train, y_train):
        """XGBoost with grid search"""
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model
    
    def train_svm(self, X_train, y_train):
        """SVM with RBF kernel"""
        svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        svm.fit(X_train, y_train)
        return svm
    
    def train_mlp(self, X_train, y_train):
        """Multi-layer Perceptron"""
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train, y_train)
        return mlp
```

#### CNN Model (EXACT SPECIFICATION)

```python
# src/ml/cnn_model.py

import tensorflow as tf
from tensorflow import keras

def build_cnn_model():
    """
    Build the CNN architecture specified in Protocol Section 6.3.3
    This is the EXACT architecture - do not modify
    """
    model = keras.Sequential([
        # Layer 1
        keras.layers.Conv1D(32, kernel_size=16, strides=2, activation='relu', 
                           input_shape=(1024, 1)),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # Layer 2
        keras.layers.Conv1D(64, kernel_size=8, strides=2, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # Layer 3
        keras.layers.Conv1D(128, kernel_size=4, strides=1, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Train CNN with early stopping as specified
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history
```

### 5. Model Evaluation

```python
# src/ml/evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class ModelEvaluator:
    """
    Evaluate and compare the 5 models
    """
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Calculate all metrics from Protocol Section 6.3.4
        """
        y_pred = model.predict(X_test)
        if len(y_pred.shape) > 1:  # CNN outputs probabilities
            y_pred = np.argmax(y_pred, axis=1)
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def compare_models(self, models, X_test, y_test):
        """
        Create comparison table (Protocol Section 6.3.4)
        """
        results = []
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results.append(metrics)
        
        return pd.DataFrame(results)
```

### 6. SiPM Characterization (src/sipm/)

Implement methods from Protocol Section 6.4:

**Crosstalk (6.4.1):**
- Single photoelectron spectrum analysis
- Energy-dependent crosstalk measurement
- Expected: 5-30% depending on scintillator

**Afterpulsing (6.4.2):**
- Inter-event time distribution analysis
- Autocorrelation method
- Expected: 1-5% probability

**Saturation (6.4.3):**
- Nonlinearity fitting: N_fired = N_cells * (1 - exp(-N_photons/N_cells))
- Identify saturation onset energy
- Correction algorithm

### 7. Pile-up Handling (src/pileup/)

From Protocol Section 6.5:

**Detection Methods:**
- Baseline restoration check
- Pulse shape fitting quality (chi-square)
- Second derivative test
- ML-based (use one of the trained classifiers)

**Correction:**
- Deconvolution algorithm (exponential subtraction)
- Scintillator-specific parameters (decay times: 2.4, 40, 230, 300 ns)

---

## 📓 Jupyter Notebooks (AS SPECIFIED)

### Notebook 4: ML Classification

**Follow the specification in `Jupyter_Notebook_Specifications.md` Section 4 EXACTLY.**

Structure (14 sections as specified):
1. Setup and Data Preparation
2. Load Feature Dataset
3. Train-Val-Test Split (70-15-15)
4. Baseline: Random Forest
5. XGBoost Classifier
6. Support Vector Machine
7. Neural Network (MLP)
8. Model Comparison (4 feature-based models)
9. Deep Learning: CNN on Raw Waveforms
   - 9.1: Prepare waveform dataset
   - 9.2: Build CNN (exact architecture)
   - 9.3: Train CNN
   - 9.4: Evaluate CNN
10. Robustness Testing (energy dependence, noise)
11. Model Interpretability (SHAP for feature-based, saliency for CNN)
12. Real-Time Classification Demo
13. Model Deployment (save models)
14. Summary and Recommendations

**Expected outcome:** 
- 4 feature-based models: 92-97% accuracy
- CNN: 95-99% accuracy
- Total: 5 models, comprehensive comparison

This is **exactly** what delivers the "ML Classification" paper from the protocol.

---

## 🎯 Success Criteria

### From Protocol Section 3.3:

✅ **Minimum Viable Study:**
- Complete energy calibration (4 scintillators × 6 sources)
- Extract >1000 pulses per configuration for ML training
- Achieve ML classification accuracy >90%
- Quantify crosstalk in at least 2 scintillators
- Demonstrate pile-up correction improvement >20%

✅ **Ideal Complete Study:**
- ML accuracy >95% with cross-validation
- Full crosstalk characterization across energy range
- Real-time implementation demonstration
- 3 publishable manuscripts

---

## 📊 Expected Results (From Protocol Section 7)

### ML Classification Performance

| Model | Expected Accuracy | Type |
|-------|-------------------|------|
| Random Forest | 92-96% | Feature-based |
| XGBoost | 93-97% | Feature-based |
| SVM | 90-94% | Feature-based |
| MLP | 92-96% | Feature-based |
| **CNN** | **95-99%** | Deep learning |

### Confusion Matrix Prediction

Most errors expected between:
- LYSO ↔ Plastic (both fast decay)
- NaI ↔ BGO (both slow, similar decay times)

---

## 📚 Publication Plan (FROM PROTOCOL SECTION 8)

This codebase supports **exactly 3 papers**:

### Paper 1: Comprehensive Comparison
*"Comprehensive Characterization of LYSO, BGO, NaI(Tl), and Plastic Scintillators Coupled to Silicon Photomultipliers"*
- Target: Nuclear Instruments and Methods A
- Notebooks: 1, 2, 3, 6, 7

### Paper 2: ML Classification  
*"Machine Learning-Based Scintillator Identification from Raw SiPM Pulse Waveforms"*
- Target: IEEE Transactions on Nuclear Science
- Notebooks: 3, 4, 7

### Paper 3: Pile-up Correction
*"Scintillator-Specific Pile-up Correction Algorithms for High Count Rate SiPM-Based Gamma Spectroscopy"*
- Target: Nuclear Instruments and Methods A
- Notebooks: 5, 7

**That's it.** Three focused, high-quality papers from one coherent study.

---

## ⚠️ What NOT to Implement

**DO NOT** implement these (they are future work, not this project):

❌ Physics-Informed Neural Networks (PINNs)  
❌ Transformer models  
❌ Vision Transformers (ViT)  
❌ Wavelet Scattering Networks  
❌ Hybrid CNN+Transformer models  
❌ ResNet architectures  
❌ Attention mechanisms  
❌ Any models beyond the 5 specified  

**Why?** These are excellent ideas for **follow-up papers** after this study is complete. The current project is already ambitious enough (9-12 months, 3 papers). Adding these would make it unfeasible.

---

## 🎯 Code Quality Requirements

### Must Have:
- ✅ Type hints for all functions
- ✅ Comprehensive docstrings (Google style)
- ✅ Error handling and validation
- ✅ Logging for training progress
- ✅ Reproducibility (set random seeds: 42)
- ✅ Modular, reusable code
- ✅ Unit tests (>70% coverage goal)
- ✅ Follow specifications exactly

### Style:
- PEP 8 compliant
- Clear variable names
- Comments for complex logic
- No dead code

---

## 📋 Deliverables Checklist

- [ ] Complete `src/` module matching structure above
- [ ] 8 Jupyter notebooks following specifications exactly
- [ ] Test suite for core functions
- [ ] Configuration file (model_config.yaml)
- [ ] requirements.txt with exact versions
- [ ] README updates
- [ ] Example data loading script
- [ ] Documentation (docstrings throughout)

---

## 🚀 Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. Data loading (src/io/)
2. Basic visualization
3. Test framework setup

### Phase 2: Traditional Analysis (Week 2)
4. Energy calibration (src/calibration/)
5. Pulse shape analysis (src/pulse_analysis/)
6. Notebooks 1-3

### Phase 3: Machine Learning (Week 3)
7. Feature-based models (src/ml/feature_based.py)
8. CNN model (src/ml/cnn_model.py)
9. Evaluation framework (src/ml/evaluation.py)
10. Notebook 4

### Phase 4: Advanced Characterization (Week 4)
11. SiPM analysis (src/sipm/)
12. Pile-up correction (src/pileup/)
13. Notebooks 5-6

### Phase 5: Synthesis (Week 5)
14. Comprehensive comparison (Notebook 7)
15. Paper figures (Notebook 8)
16. Documentation and polish

---

## 🎓 This is a Complete, Focused Study

The specifications define a substantial research project:
- **4 scintillators × 6 sources** = 24 measurement configurations
- **6 analysis areas**: calibration, pulse analysis, ML, pile-up, SiPM, comparison
- **5 ML models** with comprehensive evaluation
- **8 Jupyter notebooks** (150+ pages of analysis)
- **3 high-quality papers**

This is **sufficient** for a complete PhD chapter or postdoc project. It will:
- Fill genuine gaps in the literature
- Provide valuable community resources
- Generate high-impact publications

**DO NOT expand scope.** Additional ML models are excellent candidates for **follow-up papers** that build on this foundation.

---

## 📖 Framework Choice

Use **TensorFlow/Keras** for the CNN (as shown in protocol examples).  
Use **scikit-learn** and **XGBoost** for feature-based models.  

This matches the protocol specifications and examples.

---

## ✅ Final Confirmation

This prompt implements:
- ✅ Everything in `SiPM_Detector_Study_Complete_Protocol.md`
- ✅ Everything in `Jupyter_Notebook_Specifications.md`
- ✅ Nothing extra

Generate the codebase that delivers the 3-paper study as planned. Focus on:
1. **Correctness** - Follow specifications exactly
2. **Completeness** - All specified features
3. **Quality** - Production-ready code
4. **Clarity** - Well-documented

This is the complete, coherent implementation plan. Let's build it! 🚀
