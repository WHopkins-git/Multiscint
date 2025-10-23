# Claude Code Prompt: SiPM Detector Analysis with Advanced ML

## 🎯 Mission

Generate a complete, production-ready Python codebase for SiPM scintillation detector analysis with **state-of-the-art machine learning models** including Physics-Informed Neural Networks (PINNs), Transformers, and Wavelet Networks.

## 📚 Context Documents

You have three comprehensive specification documents:
1. **SiPM_Detector_Study_Complete_Protocol.md** - Research methodology
2. **Jupyter_Notebook_Specifications.md** - Implementation details  
3. **README.md** - Project overview

## 🚀 Your Task: Advanced ML Implementation

### Core Requirements (From Specs)
✅ Traditional ML: Random Forest, XGBoost, SVM  
✅ Baseline Deep Learning: 1D CNN  
✅ Complete analysis pipeline: calibration, pulse analysis, pile-up, SiPM characterization

### NEW: Advanced ML Models (Primary Focus)

Implement and compare these cutting-edge architectures:

#### 1. **Physics-Informed Neural Networks (PINNs)** ⭐ PRIORITY
Incorporate physical constraints into loss functions:

```python
class PhysicsInformedCNN(nn.Module):
    """CNN with physics-based loss functions"""
    
    def physics_loss(self, waveform, predicted_class):
        """
        Custom loss encoding:
        - Exponential decay constraint (τ = 2.4, 40, 230, 300 ns)
        - Energy conservation (integral = amplitude)
        - Rise time consistency
        """
        # Fit exponential to tail
        decay_loss = self.decay_time_loss(waveform, predicted_class)
        
        # Check energy conservation
        energy_loss = self.energy_conservation_loss(waveform)
        
        # Combined with classification loss
        return classification_loss + α*decay_loss + β*energy_loss
```

**Key Research Questions:**
- Does physics loss improve accuracy?
- Better data efficiency (less training data needed)?
- More robust to noise/energy variations?
- Do predictions become more physically plausible?

#### 2. **Transformer Models** ⭐ PRIORITY
Apply self-attention to pulse waveforms:

```python
class WaveformTransformer(nn.Module):
    """
    Transformer for 1D time series
    - Positional encoding for temporal structure
    - Multi-head self-attention to learn important time regions
    - Compare to CNN
    """
    def __init__(self, d_model=64, nhead=8, num_layers=4):
        self.pos_encoding = PositionalEncoding(d_model, max_len=1024)
        self.transformer = nn.TransformerEncoder(...)
        
    def forward(self, x):
        # Returns: predictions + attention_weights (for visualization)
        pass

class VisionTransformerWaveform(nn.Module):
    """
    Adapt ViT for waveforms
    - Split into patches (16 samples/patch)
    - May be more efficient than per-sample attention
    """
    pass
```

**Key Research Questions:**
- Are transformers better than CNNs for pulses?
- What do attention weights reveal about pulse discrimination?
- Speed vs. accuracy trade-off?

#### 3. **Wavelet Scattering Networks**
Physics-motivated multi-scale analysis:

```python
from kymatio.torch import Scattering1D

class WaveletScatteringClassifier:
    """
    Wavelet transform + shallow classifier
    - More interpretable than deep learning
    - Captures multi-scale temporal features
    - Physics-grounded (wavelets match decay times?)
    """
    def __init__(self, J=6, Q=8):
        self.scattering = Scattering1D(J=J, Q=Q, T=1024)
```

**Key Research Questions:**
- Competitive accuracy with interpretability?
- Do important scales correlate with decay times?
- Bridge traditional signal processing and ML?

#### 4. **Hybrid Architectures**
Best of multiple worlds:

```python
class CNNTransformerHybrid(nn.Module):
    """CNN for local features + Transformer for global context"""
    
class ResNet1DWithAttention(nn.Module):
    """ResNet backbone + attention layers"""
    
class AttentionAugmentedCNN(nn.Module):
    """CNN with attention gates"""
```

### 🎯 Comprehensive Model Comparison

Implement framework to compare 9+ models across:

| Metric | Importance |
|--------|------------|
| **Accuracy** | Classification performance |
| **Inference Speed** | Real-time capability (ms/sample) |
| **Data Efficiency** | Accuracy vs. training set size |
| **Robustness** | Performance under noise, energy variations |
| **Interpretability** | Can we understand learned features? |
| **Physical Plausibility** | Do predictions match known physics? |

## 📁 Project Structure

```
sipm-analysis/
├── src/
│   ├── io/                         # Data loading
│   ├── calibration/                # Energy calibration  
│   ├── pulse_analysis/             # Feature extraction
│   ├── ml/                         # ⭐ MAIN FOCUS
│   │   ├── models.py              # Base classes
│   │   ├── traditional_ml.py      # RF, XGBoost, SVM
│   │   ├── cnn_models.py          # Baseline CNN, ResNet
│   │   ├── physics_informed.py    # ⭐ PINNs with custom loss
│   │   ├── transformer_models.py  # ⭐ Transformer, ViT
│   │   ├── wavelet_models.py      # ⭐ Wavelet scattering
│   │   ├── hybrid_models.py       # ⭐ CNN+Transformer, etc.
│   │   ├── training.py            # Training loops
│   │   ├── evaluation.py          # ⭐ Comprehensive comparison
│   │   └── interpretability.py    # ⭐ Attention viz, SHAP
│   ├── sipm/                       # Crosstalk, afterpulsing
│   ├── pileup/                     # Detection, correction
│   └── visualization/              # Plotting
├── notebooks/
│   ├── 01-03_*.ipynb              # Standard analysis (from specs)
│   ├── 04_ml_classification.ipynb # Updated with all models
│   ├── 04b_advanced_ml_comparison.ipynb  # ⭐ NEW: Deep dive
│   ├── 04c_physics_informed_analysis.ipynb  # ⭐ NEW: PINN focus
│   ├── 05-08_*.ipynb              # Remaining analysis
├── configs/
│   └── model_configs/
│       ├── pinn_config.yaml
│       ├── transformer_config.yaml
│       └── wavelet_config.yaml
├── tests/
└── scripts/
```

## 🔬 Key Implementations

### 1. Physics-Informed Loss Functions

```python
# src/ml/physics_informed.py

class PhysicsLoss(nn.Module):
    def __init__(self, decay_times={'Plastic': 2.4, 'LYSO': 40, 'NaI': 230, 'BGO': 300}):
        self.decay_times = decay_times
        
    def decay_time_loss(self, waveform, predicted_class):
        """
        Penalize if waveform decay doesn't match expected tau
        
        1. Identify peak position
        2. Fit exponential to tail: A*exp(-t/tau)
        3. Compare fitted tau to expected tau
        """
        peak_idx = torch.argmax(waveform, dim=-1)
        
        # Extract tail (from peak to end)
        tail = waveform[..., peak_idx:]
        
        # Fit exponential (differentiable)
        # Method 1: Log-linear regression
        # Method 2: Weighted least squares
        
        fitted_tau = fit_exponential_differentiable(tail)
        expected_tau = self.decay_times[predicted_class]
        
        loss = (fitted_tau - expected_tau) ** 2
        return loss.mean()
    
    def energy_conservation_loss(self, waveform, amplitude):
        """
        Total charge (integral) should equal amplitude
        """
        integrated_charge = torch.sum(waveform, dim=-1)
        loss = (integrated_charge - amplitude) ** 2
        return loss.mean()
    
    def forward(self, waveform, predicted_class, amplitude):
        decay_loss = self.decay_time_loss(waveform, predicted_class)
        energy_loss = self.energy_conservation_loss(waveform, amplitude)
        return decay_loss, energy_loss

class PhysicsInformedCNN(nn.Module):
    def __init__(self, num_classes=4, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.cnn = StandardCNN(num_classes)
        self.physics_loss = PhysicsLoss()
        self.alpha = alpha  # Classification weight
        self.beta = beta    # Decay time weight
        self.gamma = gamma  # Energy conservation weight
        
    def compute_loss(self, waveform, labels, amplitude):
        # Classification loss
        logits = self.cnn(waveform)
        ce_loss = F.cross_entropy(logits, labels)
        
        # Physics losses
        predicted_class = torch.argmax(logits, dim=1)
        decay_loss, energy_loss = self.physics_loss(waveform, predicted_class, amplitude)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + self.beta * decay_loss + self.gamma * energy_loss
        
        return total_loss, {'ce': ce_loss, 'decay': decay_loss, 'energy': energy_loss}
```

### 2. Transformer Implementation

```python
# src/ml/transformer_models.py

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:x.size(1), :]

class WaveformTransformer(nn.Module):
    def __init__(self, waveform_length=1024, d_model=64, nhead=8, 
                 num_layers=4, num_classes=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, waveform_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        # x: [batch, 1024]
        x = x.unsqueeze(-1)  # [batch, 1024, 1]
        
        # Project to d_model
        x = self.input_proj(x)  # [batch, 1024, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        if return_attention:
            # Store attention weights
            attention_weights = []
            x = self.transformer(x)  # Need to modify to return attention
        else:
            x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        return logits

class VisionTransformerWaveform(nn.Module):
    """Adapt ViT for 1D waveforms"""
    def __init__(self, waveform_length=1024, patch_size=16, d_model=64, 
                 nhead=8, num_layers=4, num_classes=4):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = waveform_length // patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: [batch, 1024]
        batch_size = x.shape[0]
        
        # Create patches
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Embed patches
        x = self.patch_embed(x)  # [batch, num_patches, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches+1, d_model]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits
```

### 3. Comprehensive Evaluation Framework

```python
# src/ml/evaluation.py

class ModelComparison:
    """Compare all models across multiple metrics"""
    
    def __init__(self, models_dict, device='cuda'):
        self.models = models_dict
        self.device = device
        
    def evaluate_all(self, test_loader):
        """Comprehensive evaluation"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            results[name] = {
                'accuracy': self.compute_accuracy(model, test_loader),
                'precision': self.compute_precision(model, test_loader),
                'recall': self.compute_recall(model, test_loader),
                'f1_score': self.compute_f1(model, test_loader),
                'inference_time_ms': self.measure_inference_speed(model),
                'model_size_mb': self.get_model_size(model),
                'confusion_matrix': self.compute_confusion_matrix(model, test_loader),
            }
            
            # Energy-dependent accuracy
            results[name]['energy_dependent'] = self.evaluate_by_energy(model, test_loader)
            
            # Noise robustness
            results[name]['noise_robustness'] = self.evaluate_noise_robustness(model, test_loader)
        
        return pd.DataFrame(results).T
    
    def data_efficiency_curve(self, models, train_dataset, test_loader, 
                            training_sizes=[100, 500, 1000, 5000, 10000]):
        """
        Plot accuracy vs. training set size
        Shows which models learn fastest
        """
        results = {name: [] for name in models.keys()}
        
        for n in training_sizes:
            print(f"\nTraining with {n} samples...")
            
            # Sample subset
            subset = Subset(train_dataset, np.random.choice(len(train_dataset), n, replace=False))
            subset_loader = DataLoader(subset, batch_size=64, shuffle=True)
            
            for name, model in models.items():
                # Reset model
                model.reset_parameters()
                
                # Train on subset
                train_model(model, subset_loader, epochs=50, verbose=False)
                
                # Evaluate
                acc = self.compute_accuracy(model, test_loader)
                results[name].append(acc)
        
        # Plot
        plt.figure(figsize=(10, 6))
        for name, accuracies in results.items():
            plt.plot(training_sizes, accuracies, 'o-', label=name, linewidth=2)
        
        plt.xlabel('Training Set Size', fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=14)
        plt.title('Data Efficiency Comparison', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        return results
    
    def visualize_comparison(self, results_df):
        """Create comprehensive comparison visualizations"""
        
        # 1. Accuracy bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].barh(results_df.index, results_df['accuracy'] * 100)
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_title('Classification Accuracy')
        
        # Speed vs. Accuracy scatter
        axes[0, 1].scatter(results_df['inference_time_ms'], 
                          results_df['accuracy'] * 100, s=100)
        for name, row in results_df.iterrows():
            axes[0, 1].annotate(name, (row['inference_time_ms'], row['accuracy']*100),
                               fontsize=8)
        axes[0, 1].set_xlabel('Inference Time (ms)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Speed-Accuracy Trade-off')
        
        # Model size vs. accuracy
        axes[1, 0].scatter(results_df['model_size_mb'], 
                          results_df['accuracy'] * 100, s=100)
        for name, row in results_df.iterrows():
            axes[1, 0].annotate(name, (row['model_size_mb'], row['accuracy']*100),
                               fontsize=8)
        axes[1, 0].set_xlabel('Model Size (MB)')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Size-Accuracy Trade-off')
        
        # Multi-metric radar chart
        self.plot_radar_chart(results_df, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
```

### 4. Interpretability Tools

```python
# src/ml/interpretability.py

class ModelInterpretability:
    """Understand what models learned"""
    
    def visualize_transformer_attention(self, model, waveform):
        """
        Visualize attention weights over time
        Shows which parts of pulse the model focuses on
        """
        model.eval()
        with torch.no_grad():
            logits, attention_weights = model(waveform.unsqueeze(0), 
                                             return_attention=True)
        
        # Plot attention heatmap
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        axes[0].plot(waveform.cpu().numpy())
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Input Waveform')
        axes[0].grid(True, alpha=0.3)
        
        # Attention weights
        attention = attention_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        # Average over heads
        avg_attention = attention.mean(axis=0)
        
        im = axes[1].imshow(avg_attention, aspect='auto', cmap='hot', 
                           origin='lower', interpolation='nearest')
        axes[1].set_xlabel('Time Sample')
        axes[1].set_ylabel('Time Sample')
        axes[1].set_title('Attention Weights (averaged over heads)')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def compare_physics_learned_vs_known(self, pinn_model, test_data):
        """
        For physics-informed model:
        Extract learned decay times and compare to known values
        """
        known_decay_times = {'Plastic': 2.4, 'LYSO': 40, 'NaI': 230, 'BGO': 300}
        learned_decay_times = {}
        
        for scintillator, decay_time in known_decay_times.items():
            # Get pulses for this scintillator
            scint_pulses = [p for p in test_data if p.scintillator == scintillator]
            
            # Extract learned decay time from model
            # (Fit exponentials to model's internal representations)
            learned_tau = extract_learned_decay_time(pinn_model, scint_pulses)
            learned_decay_times[scintillator] = learned_tau
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(8, 8))
        
        scints = list(known_decay_times.keys())
        known = [known_decay_times[s] for s in scints]
        learned = [learned_decay_times[s] for s in scints]
        
        ax.scatter(known, learned, s=200, alpha=0.7)
        for scint, k, l in zip(scints, known, learned):
            ax.annotate(scint, (k, l), fontsize=12)
        
        # Perfect match line
        max_val = max(max(known), max(learned))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax.set_xlabel('Known Decay Time (ns)', fontsize=14)
        ax.set_ylabel('Learned Decay Time (ns)', fontsize=14)
        ax.set_title('Physics-Informed Model: Learned vs. Known', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, learned_decay_times
```

## 📓 New Notebooks

### Notebook 04b: Advanced ML Comparison

**Structure:**
1. Load data and models
2. Train all advanced models (PINN, Transformer, Wavelet, Hybrids)
3. Comprehensive comparison (9+ models)
4. Data efficiency curves
5. Interpretability analysis
6. Recommendations

### Notebook 04c: Physics-Informed Deep Dive

**Structure:**
1. Explain physics constraints mathematically
2. Implement custom loss functions
3. Ablation study (impact of each physics term)
4. Compare PINN vs. pure CNN
5. Validate learned physics
6. Generalization tests

## 🎯 Key Research Questions to Answer

1. **Do physics constraints improve ML models?**
   - Accuracy improvement?
   - Better data efficiency?
   - More robust generalization?

2. **Are transformers better than CNNs for waveforms?**
   - Classification performance?
   - Interpretability via attention?
   - Speed-accuracy trade-off?

3. **Can interpretable methods (wavelets) match black-box performance?**
   - Accuracy competitive with CNN?
   - Do important scales match physics?

4. **What's the practical recommendation?**
   - Real-time applications → which model?
   - Limited data → which model?
   - Need interpretability → which model?

## ✅ Deliverables Checklist

- [ ] Complete `src/ml/` module with 7 files
- [ ] Traditional ML baseline (RF, XGBoost, SVM)
- [ ] Baseline CNN implementation
- [ ] Physics-Informed CNN with custom loss
- [ ] Transformer models (standard + ViT)
- [ ] Wavelet scattering network
- [ ] Hybrid models (CNN+Transformer, etc.)
- [ ] Comprehensive evaluation framework
- [ ] Interpretability tools (attention viz, SHAP, saliency)
- [ ] 10 Jupyter notebooks (8 original + 2 new)
- [ ] Model comparison results (table + visualizations)
- [ ] Configuration files for all models
- [ ] Unit tests for ML modules
- [ ] Documentation and README updates

## 🚀 Start Here

**Priority Order:**
1. ✅ Basic infrastructure (data loading, training loop)
2. ✅ Traditional ML baselines
3. ✅ Baseline CNN
4. ⭐ Physics-Informed CNN (highest novelty)
5. ⭐ Transformer models
6. ⭐ Evaluation framework
7. ⭐ Wavelet networks
8. ⭐ Hybrid models
9. ⭐ Interpretability tools
10. ✅ Notebooks 04b and 04c

## 💡 Implementation Tips

### Use PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

### Set Seeds for Reproducibility
```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Efficient Training
```python
# Use mixed precision for speed
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in train_loader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Save Best Models
```python
best_acc = 0
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'accuracy': val_acc,
            'config': config
        }, 'best_model.pt')
```

## 📊 Expected Results

### Model Comparison Table (Goal)

| Model | Accuracy | Inference (ms) | Parameters | Data Efficiency |
|-------|----------|----------------|------------|-----------------|
| Random Forest | 94% | 2 | N/A | ⭐⭐⭐ |
| XGBoost | 97% | 3 | N/A | ⭐⭐⭐⭐ |
| CNN Baseline | 96% | 1.5 | 500K | ⭐⭐⭐ |
| **PINN** | **97%** | 1.5 | 500K | ⭐⭐⭐⭐⭐ |
| Transformer | 98% | 3 | 2M | ⭐⭐ |
| ViT | 97% | 2.5 | 1.5M | ⭐⭐⭐ |
| Wavelet + SVM | 95% | 2 | 10K | ⭐⭐⭐⭐ |
| CNN+Transformer | **98.5%** | 2 | 1M | ⭐⭐⭐ |

## 🎓 Publication Impact

This comprehensive ML comparison will result in **2-3 additional papers**:

1. **"Physics-Informed Deep Learning for Scintillation Pulse Classification"**
   - IEEE Trans. Nuclear Science (high impact)
   - Novel PINN application in instrumentation

2. **"Transformer-Based Real-Time Scintillator Identification"**
   - Nuclear Instruments and Methods A
   - State-of-the-art accuracy + interpretability

3. **"Comprehensive Machine Learning Benchmark for Radiation Detector Analysis"**
   - Journal of Instrumentation
   - Community resource

---

## 🚀 Ready to Generate!

Please create the complete codebase following these specifications. Focus on:
1. ✅ **Correctness** - Working implementations
2. ✅ **Clarity** - Well-documented code
3. ✅ **Completeness** - All features included
4. ✅ **Novelty** - Cutting-edge ML methods

Generate production-ready Python code with full documentation, type hints, error handling, and tests. Make this the definitive open-source framework for ML-based scintillator analysis! 🎯
