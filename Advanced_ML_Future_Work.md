# Advanced Machine Learning for Scintillator Classification: Future Research Roadmap

## 📋 Document Purpose

This document outlines **future research directions** in advanced machine learning for SiPM-scintillator pulse classification. These are **not part of the current 3-paper study**, but represent logical next steps after the foundational work is complete.

**Status:** Future Work / Papers 4-6  
**Prerequisites:** Completion of baseline study (Papers 1-3) with traditional ML and CNN baseline  
**Timeline:** 12-18 months after baseline completion  
**Target Journals:** IEEE TNS, NIM A, Machine Learning: Science and Technology

---

## 🎯 Executive Summary

The baseline study establishes a comprehensive dataset and achieves 95-99% accuracy with traditional ML and CNNs. This future work explores:

1. **Physics-Informed Neural Networks (PINNs)** - Incorporating known scintillator physics into loss functions
2. **Transformer Models** - Self-attention mechanisms for temporal pulse analysis
3. **Wavelet Scattering Networks** - Multi-scale, interpretable feature extraction
4. **Hybrid Architectures** - Combining strengths of multiple approaches

**Research Questions:**
- Can physics constraints improve data efficiency and generalization?
- Do transformers capture temporal dependencies better than CNNs?
- Can interpretable methods (wavelets) match black-box performance?
- What is the optimal architecture for real-time deployment?

**Expected Outcome:** 2-3 additional high-impact papers advancing ML in radiation detection

---

## Table of Contents

1. [Motivation and Context](#motivation)
2. [Physics-Informed Neural Networks (PINNs)](#pinns)
3. [Transformer Models](#transformers)
4. [Wavelet Scattering Networks](#wavelets)
5. [Hybrid Architectures](#hybrids)
6. [Experimental Design](#experimental-design)
7. [Expected Results](#expected-results)
8. [Publication Strategy](#publication-strategy)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Technical Challenges and Mitigation](#challenges)

---

## 1. Motivation and Context {#motivation}

### 1.1 Why Advanced ML After Baseline?

The baseline study (Papers 1-3) will establish:
- ✅ Comprehensive benchmark dataset (4 scintillators, 6 sources, 40k+ waveforms)
- ✅ Validated data collection pipeline
- ✅ Traditional ML baseline (RF, XGBoost: 92-97% accuracy)
- ✅ CNN baseline (95-99% accuracy)
- ✅ Understanding of problem difficulty and failure modes

With this foundation, we can explore:
- **Data efficiency**: Can we match CNN accuracy with less training data?
- **Interpretability**: What features do models learn? Do they match physics?
- **Generalization**: Can models handle new scintillators or energies?
- **Real-time performance**: Speed vs. accuracy trade-offs for deployment

### 1.2 Novelty in Nuclear Instrumentation

**Current State:**
- CNNs applied to pulse shape discrimination (neutron/gamma in organic scintillators)
- Traditional ML for scintillator identification
- Limited exploration of modern architectures (transformers, attention)

**Gaps:**
- No physics-informed approaches in scintillation detection
- No transformer applications to pulse waveforms
- No systematic comparison of architectures
- No interpretability analysis linking ML to scintillator physics

**Our Contribution:**
- First PINN application to scintillator classification
- First transformer-based pulse shape analysis
- Comprehensive architectural comparison
- Bridge between ML and radiation physics

### 1.3 Research Impact

**Scientific:**
- Advance understanding of what makes pulses discriminable
- Validate that ML can learn known physics
- Establish best practices for ML in instrumentation

**Practical:**
- Enable real-time detector identification in field deployments
- Reduce training data requirements (important for rare isotopes)
- Provide interpretable models for safety-critical applications

**Community:**
- Open-source implementations
- Benchmark dataset for ML researchers
- Tutorial notebooks for practitioners

---

## 2. Physics-Informed Neural Networks (PINNs) {#pinns}

### 2.1 Core Concept

**Traditional ML:** Learn patterns purely from data  
**Physics-Informed ML:** Constrain learning with known physical laws

For scintillators, we know:
1. **Exponential Decay:** Pulse tail follows $I(t) = I_0 e^{-t/\tau}$ where $\tau$ is scintillator-specific
2. **Charge Conservation:** Total integrated charge proportional to energy deposited
3. **Rise Time Constraints:** Limited by scintillator and SiPM response
4. **Amplitude-Energy Relationship:** Linear for single scintillator

### 2.2 Architecture Design

#### 2.2.1 Multi-Task PINN

```python
class PhysicsInformedCNN(nn.Module):
    """
    CNN that predicts both class and physical parameters
    """
    def __init__(self):
        # Shared feature extractor
        self.conv_layers = ConvolutionalBackbone()
        
        # Task heads
        self.classifier = nn.Linear(128, 4)  # 4 scintillators
        self.decay_predictor = nn.Linear(128, 1)  # Predict τ
        self.rise_predictor = nn.Linear(128, 1)  # Predict rise time
        
    def forward(self, x):
        features = self.conv_layers(x)
        
        class_logits = self.classifier(features)
        decay_time = self.decay_predictor(features)
        rise_time = self.rise_predictor(features)
        
        return class_logits, decay_time, rise_time
```

#### 2.2.2 Physics-Based Loss Function

**Multi-Task Loss:**
```python
def physics_informed_loss(pred_class, pred_tau, pred_rise, 
                         true_class, waveform, 
                         alpha=0.6, beta=0.3, gamma=0.1):
    """
    Combined loss: Classification + Physics constraints
    
    Args:
        alpha: Classification weight
        beta: Decay time physics weight
        gamma: Rise time physics weight
    """
    # Standard classification loss
    ce_loss = F.cross_entropy(pred_class, true_class)
    
    # Physics loss 1: Decay time consistency
    # Known decay times: Plastic=2.4ns, LYSO=40ns, NaI=230ns, BGO=300ns
    true_decay_times = torch.tensor([2.4, 40, 230, 300])[true_class]
    decay_loss = F.mse_loss(pred_tau, true_decay_times)
    
    # Physics loss 2: Rise time consistency
    # Known rise times: Plastic=0.5ns, LYSO=0.1ns, NaI=1ns, BGO=2ns
    true_rise_times = torch.tensor([0.5, 0.1, 1.0, 2.0])[true_class]
    rise_loss = F.mse_loss(pred_rise, true_rise_times)
    
    # Combined loss
    total_loss = alpha * ce_loss + beta * decay_loss + gamma * rise_loss
    
    return total_loss, {
        'classification': ce_loss,
        'decay': decay_loss,
        'rise': rise_loss
    }
```

**Key Design Principle:** Physics losses use **true_class**, not predicted class, avoiding the early-training catch-22 identified by Gemini.

#### 2.2.3 Alternative: Soft Physics Regularization

```python
def soft_physics_regularization(waveform, pred_class, pred_tau):
    """
    Regularization term penalizing physically implausible predictions
    
    Instead of hard constraints, use soft penalties that guide learning
    """
    # Extract tail of waveform
    peak_idx = torch.argmax(waveform, dim=-1)
    tail = waveform[:, peak_idx:]
    
    # Fit exponential to tail (simple log-linear regression)
    time = torch.arange(tail.shape[1]).float()
    log_tail = torch.log(tail + 1e-8)  # Add epsilon for stability
    
    # Linear fit: log(I) = log(I0) - t/τ
    # Extracted τ should be close to predicted τ
    fitted_tau = fit_tau_least_squares(time, log_tail)
    
    # Soft penalty: Allow some deviation but penalize large differences
    penalty = torch.abs(fitted_tau - pred_tau) / (pred_tau + 1e-8)
    
    return penalty.mean()
```

### 2.3 Research Questions

**RQ1: Data Efficiency**
- Hypothesis: PINN will achieve CNN baseline accuracy with 50% less training data
- Test: Train on [100, 500, 1000, 5000, 10000] samples, compare to baseline CNN
- Metric: Training set size to reach 95% accuracy

**RQ2: Generalization**
- Hypothesis: Physics constraints improve generalization to new energies/conditions
- Test: Train on limited energy range, test on full range
- Metric: Accuracy degradation vs. baseline

**RQ3: Physical Plausibility**
- Hypothesis: PINN predictions match known scintillator properties
- Test: Compare predicted τ and rise times to literature values
- Metric: Relative error in physical parameters

**RQ4: Interpretability**
- Hypothesis: PINN internal representations correlate with physics
- Test: Analyze learned features vs. decay/rise times
- Metric: Correlation coefficient

### 2.4 Ablation Studies

Systematic removal of components to understand contributions:

1. **Baseline:** Pure CNN (no physics)
2. **PINN-Decay:** Add decay time prediction head
3. **PINN-Rise:** Add rise time prediction head
4. **PINN-Full:** Both decay and rise time heads
5. **PINN-Reg:** Add soft physics regularization

Compare:
- Classification accuracy
- Training convergence speed
- Data efficiency
- Generalization performance

### 2.5 Expected Results

| Model | Accuracy | Data to 95% | Generalization | Interpret. |
|-------|----------|-------------|----------------|------------|
| Baseline CNN | 97% | 10k samples | Good | Low |
| PINN-Decay | 97% | 7k samples | Better | Medium |
| PINN-Full | 97.5% | 5k samples | Better | High |
| PINN-Reg | 97.8% | 4k samples | Best | High |

**Key Finding:** Physics constraints reduce data requirements by 50-60% while improving interpretability.

### 2.6 Technical Challenges

**Challenge 1: Differentiable Fitting**
- Problem: Exponential fitting in loss function must be differentiable
- Solution: Use log-linear regression or pre-compute on CPU
- Alternative: Learn τ as auxiliary output instead of fitting

**Challenge 2: Loss Weighting**
- Problem: Balancing classification vs. physics losses
- Solution: Grid search over α, β, γ weights
- Heuristic: Start with α=0.7, β=0.2, γ=0.1

**Challenge 3: Noisy Data**
- Problem: Real waveforms have noise; exponential fits may be unstable
- Solution: Robust regression techniques, outlier rejection
- Alternative: Use exponential fit quality as uncertainty estimate

---

## 3. Transformer Models {#transformers}

### 3.1 Core Concept

**CNNs:** Process waveforms with local receptive fields (convolutional kernels)  
**Transformers:** Use self-attention to capture global temporal dependencies

**Potential Advantages:**
- Can learn long-range dependencies (e.g., correlation between pulse peak and distant tail)
- Attention weights provide interpretability (which time points matter?)
- State-of-the-art in sequence modeling (NLP, time series)

### 3.2 Architecture Design

#### 3.2.1 Standard Transformer for Waveforms

```python
class WaveformTransformer(nn.Module):
    """
    Transformer encoder for 1D waveform classification
    """
    def __init__(self, 
                 waveform_length=1024,
                 d_model=64,
                 nhead=8,
                 num_layers=4,
                 num_classes=4):
        super().__init__()
        
        # Input projection: 1D waveform → d_model embedding
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding (sine/cosine)
        self.pos_encoder = PositionalEncoding(d_model, max_len=waveform_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        # x: [batch, 1024]
        batch_size = x.shape[0]
        
        # Project to embedding space
        x = x.unsqueeze(-1)  # [batch, 1024, 1]
        x = self.input_proj(x)  # [batch, 1024, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if return_attention:
            # Extract attention weights for visualization
            attention_weights = []
            for layer in self.transformer.layers:
                x, attn = layer(x, return_attention=True)
                attention_weights.append(attn)
            transformer_output = x
        else:
            transformer_output = self.transformer(x)
        
        # Global average pooling over sequence
        pooled = transformer_output.mean(dim=1)  # [batch, d_model]
        
        # Classification
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, attention_weights
        return logits
```

#### 3.2.2 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal information
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
```

#### 3.2.3 Vision Transformer (ViT) for Waveforms

```python
class VisionTransformerWaveform(nn.Module):
    """
    Adapt Vision Transformer for 1D waveforms
    
    Key idea: Split waveform into patches (e.g., 16 samples/patch)
    Trade-off: More efficient but less fine-grained temporal resolution
    """
    def __init__(self,
                 waveform_length=1024,
                 patch_size=16,
                 d_model=64,
                 nhead=8,
                 num_layers=4,
                 num_classes=4):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = waveform_length // patch_size
        
        # Patch embedding: Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # CLS token (for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head (uses CLS token)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [batch, 1024]
        batch_size = x.shape[0]
        
        # Create patches: [batch, 1024] → [batch, 64, 16]
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Embed patches: [batch, 64, 16] → [batch, 64, d_model]
        x = self.patch_embed(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 65, d_model]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # [batch, d_model]
        logits = self.classifier(cls_output)
        
        return logits
```

### 3.3 Research Questions

**RQ1: Performance vs. CNN**
- Hypothesis: Transformers achieve comparable accuracy to CNN
- Test: Train both on same data, compare accuracy
- Metric: Top-1 accuracy, confusion matrices

**RQ2: What Does Attention Learn?**
- Hypothesis: Attention focuses on discriminative temporal regions (peaks, decay)
- Test: Visualize attention weights, correlate with pulse features
- Metric: Attention entropy, peak overlap with physical features

**RQ3: Long-Range Dependencies**
- Hypothesis: Transformers better capture peak-tail correlations
- Test: Mask different temporal regions, measure impact
- Metric: Accuracy with truncated waveforms

**RQ4: Computational Efficiency**
- Hypothesis: ViT more efficient than standard transformer
- Test: Measure training time, inference latency, memory usage
- Metric: ms/sample, FLOPS, GPU memory

### 3.4 Attention Visualization

```python
def visualize_attention(model, waveform, scintillator_name):
    """
    Visualize which parts of waveform the model attends to
    """
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(
            waveform.unsqueeze(0), 
            return_attention=True
        )
    
    # Average attention over heads and layers
    # attention_weights: List of [batch, nhead, seq_len, seq_len]
    attention = torch.stack(attention_weights).mean(dim=(0, 1, 2))
    attention = attention.cpu().numpy()  # [seq_len,]
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Waveform
    time_ns = np.arange(1024) * 8  # 8 ns sampling
    axes[0].plot(time_ns, waveform.cpu().numpy())
    axes[0].set_ylabel('Amplitude (ADC)')
    axes[0].set_title(f'{scintillator_name} Pulse Waveform')
    axes[0].grid(True, alpha=0.3)
    
    # Attention weights
    axes[1].plot(time_ns, attention)
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Transformer Attention (averaged over heads/layers)')
    axes[1].grid(True, alpha=0.3)
    
    # Overlay
    ax2 = axes[2].twinx()
    axes[2].plot(time_ns, waveform.cpu().numpy(), 'b-', alpha=0.7, label='Waveform')
    ax2.plot(time_ns, attention, 'r-', alpha=0.7, label='Attention')
    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Amplitude', color='b')
    ax2.set_ylabel('Attention', color='r')
    axes[2].set_title('Waveform + Attention Overlay')
    
    plt.tight_layout()
    return fig
```

### 3.5 Expected Results

| Model | Accuracy | Inference (ms) | Parameters | Interpretability |
|-------|----------|----------------|------------|------------------|
| CNN Baseline | 97% | 1.5 | 500K | Low |
| Transformer | 97-98% | 3-4 | 2M | **High** (attention) |
| ViT | 96-97% | 2-3 | 1.5M | **High** (attention) |

**Key Finding:** Transformers match CNN accuracy with interpretable attention, at cost of increased computation.

### 3.6 Analysis: What Makes Pulses Discriminable?

Use attention to understand physical basis of discrimination:

1. **LYSO vs. Plastic:** Fast decay (40 ns vs. 2.4 ns)
   - Expected: Attention on early tail region (0-200 ns)
   
2. **NaI vs. BGO:** Both slow (230 ns vs. 300 ns)
   - Expected: Attention on far tail (500-2000 ns) or peak amplitude

3. **All classes:** Different rise times
   - Expected: Attention on rising edge (pre-peak)

**Hypothesis:** If attention correlates with known discriminative features, this validates that model learned meaningful physics.

---

## 4. Wavelet Scattering Networks {#wavelets}

### 4.1 Core Concept

**Problem with CNNs:** Learned filters may not correspond to interpretable physical features

**Wavelet Solution:** Use predetermined, mathematically-defined filters at multiple scales
- **Interpretable:** Each scale corresponds to specific frequency/time scale
- **Physics-motivated:** Scales can be chosen to match scintillator decay times
- **Efficient:** No backpropagation through wavelet layers

### 4.2 Wavelet Scattering Transform

Based on Mallat's scattering transform (kymatio library):

```python
from kymatio.torch import Scattering1D

class WaveletScatteringClassifier:
    """
    Wavelet scattering features + simple classifier
    
    Scattering transform: Multi-scale wavelet decomposition
    - 1st order: |x * ψ_λ|
    - 2nd order: ||x * ψ_λ1| * ψ_λ2|
    
    Advantages:
    - Translation invariant
    - Captures multi-scale structure
    - Predetermined filters (no learning)
    """
    def __init__(self, J=6, Q=8, T=1024):
        """
        Args:
            J: Maximum scale (2^J samples)
            Q: Number of wavelets per octave
            T: Signal length
        """
        self.scattering = Scattering1D(J=J, Q=Q, T=T)
        self.classifier = None
        
    def extract_features(self, waveforms):
        """
        Extract scattering coefficients
        
        Args:
            waveforms: [N, 1024] tensor
            
        Returns:
            features: [N, n_features] scattering coefficients
        """
        # Ensure proper shape for scattering
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)  # [N, 1, 1024]
        
        # Apply scattering transform
        scattering_coeffs = self.scattering(waveforms)
        
        # Flatten to feature vector
        features = scattering_coeffs.view(waveforms.shape[0], -1)
        
        return features
    
    def train_classifier(self, X_train, y_train, classifier_type='svm'):
        """
        Train classifier on scattering features
        
        Scattering coefficients are not learned - only classifier weights
        """
        # Extract features
        features_train = self.extract_features(X_train)
        
        # Train classifier
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', C=10, gamma='scale')
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100)
        elif classifier_type == 'mlp':
            self.classifier = MLPClassifier(hidden_layer_sizes=(128, 64))
        
        self.classifier.fit(features_train.cpu().numpy(), y_train)
        
    def predict(self, X_test):
        """Predict classes for test waveforms"""
        features_test = self.extract_features(X_test)
        return self.classifier.predict(features_test.cpu().numpy())
```

### 4.3 Scale Selection and Physics

**Key Insight:** Wavelet scales correspond to time/frequency scales

**Scintillator Decay Times:**
- Plastic: 2.4 ns → High frequency (1/2.4 ns ≈ 400 MHz)
- LYSO: 40 ns → Medium frequency (1/40 ns ≈ 25 MHz)
- NaI: 230 ns → Low frequency (1/230 ns ≈ 4 MHz)
- BGO: 300 ns → Low frequency (1/300 ns ≈ 3 MHz)

**Hypothesis:** Wavelet coefficients at scales matching decay times will be most discriminative

**Analysis:**
```python
def analyze_scale_importance(wavelet_model, scintillator_data):
    """
    Determine which wavelet scales are most important for discrimination
    """
    # Extract scattering coefficients
    all_features = []
    all_labels = []
    
    for scint_name, waveforms in scintillator_data.items():
        features = wavelet_model.extract_features(waveforms)
        all_features.append(features)
        all_labels.extend([scint_name] * len(waveforms))
    
    all_features = torch.cat(all_features, dim=0)
    
    # Analyze feature importance (if using RF classifier)
    if hasattr(wavelet_model.classifier, 'feature_importances_'):
        importances = wavelet_model.classifier.feature_importances_
        
        # Map back to scales
        # Scattering1D returns [batch, n_scales, n_times]
        # Group importances by scale
        
        return importances
    
    # Or use mutual information
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(all_features.cpu().numpy(), all_labels)
    
    return mi
```

### 4.4 Research Questions

**RQ1: Competitive Accuracy**
- Hypothesis: Wavelets achieve 90-95% accuracy (close to CNN)
- Test: Compare on same train/test split
- Metric: Classification accuracy

**RQ2: Physics Correlation**
- Hypothesis: Important scales correlate with decay times
- Test: Feature importance analysis, compare to known τ values
- Metric: Correlation coefficient between important scales and 1/τ

**RQ3: Interpretability**
- Hypothesis: Wavelet features more interpretable than CNN features
- Test: Visualize important coefficients, link to physical features
- Metric: Qualitative assessment by domain experts

**RQ4: Computational Efficiency**
- Hypothesis: Faster than CNN (no backprop through wavelet layers)
- Test: Measure training time
- Metric: Training time, number of learnable parameters

### 4.5 Expected Results

| Model | Accuracy | Training Time | Parameters | Interpretability |
|-------|----------|---------------|------------|------------------|
| CNN | 97% | 2 hours | 500K (all learned) | Low |
| Wavelet+SVM | 93-95% | 30 min | 10K (classifier only) | **Very High** |
| Wavelet+MLP | 94-96% | 45 min | 50K (classifier only) | **Very High** |

**Key Finding:** Slight accuracy trade-off for massive gains in interpretability and efficiency.

### 4.6 Interpretability Analysis

**Visualization 1: Scattering Coefficients**
```python
def visualize_scattering_coefficients(model, waveform, scintillator):
    """
    Show which wavelet scales activate for given scintillator
    """
    coeffs = model.scattering(waveform.unsqueeze(0).unsqueeze(1))
    # coeffs: [1, n_coeffs, n_times]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original waveform
    axes[0].plot(waveform.cpu().numpy())
    axes[0].set_title(f'{scintillator} Waveform')
    
    # 1st order coefficients (different scales)
    axes[1].imshow(coeffs[0, :J*Q, :].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[1].set_title('1st Order Scattering Coefficients (by scale)')
    axes[1].set_ylabel('Scale')
    
    # 2nd order coefficients
    axes[2].imshow(coeffs[0, J*Q:, :].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[2].set_title('2nd Order Scattering Coefficients')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Scale Pair')
    
    plt.tight_layout()
    return fig
```

**Visualization 2: Scale Importance vs. Decay Time**
```python
def plot_scale_vs_decay_time(feature_importance, decay_times):
    """
    Show correlation between important scales and scintillator physics
    """
    # Convert scales to equivalent time constants
    scale_time_constants = compute_time_constants_from_scales(scales)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot feature importance vs. scale time constant
    ax.bar(scale_time_constants, feature_importance)
    ax.set_xlabel('Wavelet Scale Time Constant (ns)')
    ax.set_ylabel('Feature Importance')
    ax.set_xscale('log')
    
    # Overlay known decay times
    for scint, tau in decay_times.items():
        ax.axvline(tau, color='red', linestyle='--', alpha=0.5)
        ax.text(tau, max(feature_importance)*0.9, scint, rotation=90)
    
    ax.set_title('Wavelet Scale Importance vs. Scintillator Decay Times')
    plt.tight_layout()
    return fig
```

---

## 5. Hybrid Architectures {#hybrids}

### 5.1 Motivation

**Single Architecture Limitations:**
- CNNs: Good local features, poor long-range dependencies
- Transformers: Good global context, computationally expensive
- Wavelets: Interpretable, but fixed features

**Hybrid Solution:** Combine strengths of multiple approaches

### 5.2 CNN + Transformer Hybrid

```python
class CNNTransformerHybrid(nn.Module):
    """
    CNN for local feature extraction + Transformer for global context
    
    Architecture:
    1. CNN extracts local features (reduces sequence length)
    2. Transformer captures global dependencies
    3. Classification head
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # CNN feature extractor (reduce 1024 → 128 length)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Output: [batch, 128, 32]
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(128, max_len=32)
        
        # Transformer for global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, 1024]
        
        # CNN feature extraction
        x = x.unsqueeze(1)  # [batch, 1, 1024]
        features = self.conv_layers(x)  # [batch, 128, 32]
        
        # Transpose for transformer
        features = features.transpose(1, 2)  # [batch, 32, 128]
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        global_features = self.transformer(features)
        
        # Global pooling
        pooled = global_features.mean(dim=1)  # [batch, 128]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
```

**Advantages:**
- CNN reduces sequence length (efficient for transformer)
- Transformer captures long-range dependencies
- Best of both worlds

### 5.3 ResNet-1D with Attention

```python
class ResNet1DWithAttention(nn.Module):
    """
    ResNet-style architecture with attention gates
    
    Key features:
    - Residual connections for deep networks
    - Attention gates to focus on important features
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.attention1 = AttentionGate(64)
        
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.attention2 = AttentionGate(128)
        
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.attention3 = AttentionGate(256)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, 1024]
        x = x.unsqueeze(1)  # [batch, 1, 1024]
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.fc(x)
        
        return logits

class AttentionGate(nn.Module):
    """
    Attention gate to emphasize important features
    """
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights
```

### 5.4 Wavelet + Deep Learning

```python
class WaveletDeepHybrid(nn.Module):
    """
    Wavelet features + learned deep features
    
    Combines:
    - Fixed wavelet features (interpretable)
    - Learned CNN features (flexible)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Wavelet scattering (fixed)
        self.scattering = Scattering1D(J=6, Q=8, T=1024)
        n_wavelet_features = self._get_scattering_size()
        
        # Learnable CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_wavelet_features + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Wavelet branch
        wavelet_features = self.scattering(x.unsqueeze(1))
        wavelet_features = wavelet_features.view(x.size(0), -1)
        
        # CNN branch
        cnn_features = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_features.view(x.size(0), -1)
        
        # Concatenate and fuse
        combined = torch.cat([wavelet_features, cnn_features], dim=1)
        logits = self.fusion(combined)
        
        return logits
```

### 5.5 Expected Results

| Hybrid Model | Accuracy | Inference (ms) | Interpretability |
|--------------|----------|----------------|------------------|
| CNN+Transformer | 98% | 2.5 | Medium |
| ResNet+Attention | 97.5% | 2.0 | Medium |
| Wavelet+CNN | 96% | 1.8 | High |

**Key Finding:** Hybrids can achieve best accuracy while maintaining some interpretability.

---

## 6. Experimental Design {#experimental-design}

### 6.1 Dataset

**Use baseline study dataset:**
- 4 scintillators (LYSO, BGO, NaI, Plastic)
- 6 radiation sources (Cs-137, Na-22, Co-60, Co-57, Am-241, Sr-90)
- 40,000+ waveforms with ground truth labels
- Energy range: 59.5 - 1332 keV

**Train/Val/Test Split:** Same as baseline (70/15/15)

### 6.2 Evaluation Metrics

**Primary:**
1. **Classification Accuracy**
2. **F1 Score** (per class and weighted)
3. **Confusion Matrix**

**Secondary:**
4. **Inference Time** (ms/sample)
5. **Model Size** (parameters, MB)
6. **Training Time** (GPU hours)
7. **Memory Usage** (GB)

**Novel:**
8. **Data Efficiency Curve** (accuracy vs. training set size)
9. **Energy Generalization** (train on subset, test on full range)
10. **Noise Robustness** (accuracy vs. added noise level)
11. **Interpretability Score** (qualitative, physics correlation)

### 6.3 Experimental Protocol

#### 6.3.1 Baseline Reproduction
1. Reproduce CNN baseline from Paper 2
2. Verify 95-99% accuracy
3. Use as comparison benchmark

#### 6.3.2 Model Training
For each advanced model:
1. **Grid search hyperparameters** (learning rate, architecture size)
2. **Train with early stopping** (patience=20 epochs)
3. **Save best model** (validation accuracy)
4. **Evaluate on held-out test set**
5. **Repeat 5 times** with different random seeds (report mean ± std)

#### 6.3.3 Comparison Studies

**Study 1: Standard Comparison**
- Train all models on full dataset
- Compare accuracy, speed, size
- Generate comparison table and plots

**Study 2: Data Efficiency**
- Train on subsets: [100, 500, 1k, 5k, 10k, full]
- Plot accuracy vs. training size
- Identify "knee" of curve

**Study 3: Energy Generalization**
- Train on: Am-241 (59.5 keV), Cs-137 (662 keV) only
- Test on: All sources including Co-60 (1173, 1332 keV)
- Measure accuracy degradation

**Study 4: Noise Robustness**
- Add Gaussian noise: σ = [0, 0.05, 0.1, 0.2, 0.5] × signal std
- Test all models
- Plot accuracy vs. noise level

**Study 5: Interpretability Analysis**
- **PINN:** Compare predicted vs. true τ, rise times
- **Transformer:** Visualize attention, correlate with physics
- **Wavelet:** Analyze scale importance vs. decay times
- **Qualitative assessment** by domain experts

### 6.4 Statistical Analysis

**Significance Testing:**
- Paired t-test between models
- Bonferroni correction for multiple comparisons
- Report p-values for accuracy differences

**Effect Size:**
- Cohen's d for accuracy differences
- Classify: small (0.2), medium (0.5), large (0.8)

---

## 7. Expected Results {#expected-results}

### 7.1 Model Performance Summary

| Model | Accuracy | Inference (ms) | Params | Data to 95% | Energy Gen. | Noise Robust | Interpret. |
|-------|----------|----------------|--------|-------------|-------------|--------------|------------|
| **Baseline CNN** | 97% | 1.5 | 500K | 10k | Good | Good | Low |
| **PINN** | 97.5% | 1.6 | 550K | **5k** | **Better** | **Better** | **High** |
| **Transformer** | 97.5% | 3.5 | 2M | 12k | Good | Good | **High** |
| **ViT** | 96.5% | 2.8 | 1.5M | 10k | Good | Good | **High** |
| **Wavelet+SVM** | 94% | **1.0** | **10K** | 8k | Good | Medium | **V. High** |
| **Wavelet+MLP** | 95.5% | 1.2 | 50K | 7k | Good | Good | **V. High** |
| **CNN+Trans** | **98%** | 2.5 | 1M | 9k | **Better** | Good | Medium |
| **ResNet+Attn** | 97.8% | 2.0 | 800K | 8k | Good | Good | Medium |

### 7.2 Key Findings (Predicted)

**Finding 1: Physics Constraints Improve Data Efficiency**
- PINN achieves baseline accuracy with 50% less data
- Validates physics-informed learning for instrumentation

**Finding 2: Transformers Learn Interpretable Features**
- Attention correlates with discriminative temporal regions
- Focuses on peaks for fast scintillators, tails for slow

**Finding 3: Wavelets Provide Physics-ML Bridge**
- Important scales match scintillator decay times (r > 0.8)
- 94-95% accuracy competitive for many applications
- Massive interpretability gains

**Finding 4: Hybrids Achieve Best Accuracy**
- CNN+Transformer: 98% (1% improvement over baseline)
- Trade-off: 50% more compute

**Finding 5: All Models Generalize Well**
- <3% accuracy drop on unseen energy ranges
- Validates that models learn general pulse shape features

### 7.3 Visualization Gallery

Expected figures for publications:

1. **Comparison Bar Chart:** Accuracy across all models
2. **Speed-Accuracy Scatter:** Trade-off visualization
3. **Data Efficiency Curves:** All models, annotated knee points
4. **Transformer Attention Maps:** 4 panels (one per scintillator)
5. **Wavelet Scale Importance:** Overlay with decay times
6. **PINN Physics Validation:** Predicted vs. true τ scatter
7. **Confusion Matrices:** Best model vs. baseline
8. **Robustness Curves:** Accuracy vs. noise level
9. **Energy Generalization:** Train subset, test full range
10. **Multi-dimensional Radar:** All metrics on one plot

---

## 8. Publication Strategy {#publication-strategy}

### 8.1 Paper 4: Physics-Informed Deep Learning

**Title:** *"Physics-Informed Deep Learning for Scintillation Pulse Classification: Incorporating Decay Time Constraints"*

**Target Journal:** IEEE Transactions on Nuclear Science (Impact Factor: ~1.8)

**Structure:**
1. **Introduction**
   - Problem: Limited training data in experimental physics
   - Solution: Physics-informed neural networks
   - Novelty: First PINN application to scintillators

2. **Methods**
   - Multi-task PINN architecture
   - Physics loss functions (decay, rise times)
   - Training procedure

3. **Results**
   - 50% data efficiency improvement
   - Predicted τ matches true within 5%
   - Improved generalization

4. **Discussion**
   - Why physics constraints help
   - Comparison to pure data-driven
   - Applicability to other detectors

5. **Conclusions**

**Expected Impact:** High - novel methodology, practical benefits

### 8.2 Paper 5: Transformer-Based Pulse Analysis

**Title:** *"Transformer-Based Attention Mechanisms for Real-Time Scintillator Identification from Pulse Waveforms"*

**Target Journal:** Nuclear Instruments and Methods in Physics Research A (Impact Factor: ~1.5)

**Structure:**
1. **Introduction**
   - Transformers in time series
   - Application to pulse shape analysis
   - Interpretability via attention

2. **Methods**
   - Transformer architecture for waveforms
   - Positional encoding design
   - Attention visualization

3. **Results**
   - 97-98% accuracy (matches CNN)
   - Attention correlates with physics
   - Interpretability analysis

4. **Discussion**
   - What attention reveals about discrimination
   - Comparison to CNN features
   - Real-time feasibility

5. **Conclusions**

**Expected Impact:** Medium-High - state-of-the-art method, interpretability

### 8.3 Paper 6: Comprehensive ML Benchmark

**Title:** *"Comprehensive Comparison of Machine Learning Architectures for Radiation Detector Pulse Shape Analysis"*

**Target Journal:** Machine Learning: Science and Technology (Impact Factor: ~5.0) or Journal of Instrumentation

**Structure:**
1. **Introduction**
   - ML proliferation in physics
   - Need for systematic comparison
   - Radiation detection as testbed

2. **Methods**
   - Dataset description
   - 9 architectures implemented
   - Evaluation protocol

3. **Results**
   - Performance comparison (Table 7.1)
   - Data efficiency analysis
   - Interpretability assessment
   - Speed-accuracy trade-offs

4. **Discussion**
   - When to use which architecture
   - Practical recommendations
   - Future directions

5. **Conclusions**

**Expected Impact:** Very High - community resource, comprehensive

**Appendix:** Code and data availability

### 8.4 Conference Presentations

**IEEE NSS/MIC (Nuclear Science Symposium)**
- Present PINN and Transformer work
- Oral presentation preferred
- Strong ML track

**Machine Learning for Physical Sciences (NeurIPS Workshop)**
- Physics-informed learning focus
- High visibility to ML community

**ANIMMA (Advancements in Nuclear Instrumentation Measurement Methods and their Applications)**
- European conference
- Good for comprehensive comparison

---

## 9. Implementation Roadmap {#implementation-roadmap}

### 9.1 Timeline (18 Months Post-Baseline)

**Months 1-3: PINN Development**
- Month 1: Implement multi-task PINN architecture
- Month 2: Physics loss function design and testing
- Month 3: Ablation studies, hyperparameter tuning

**Months 4-6: Transformer Development**
- Month 4: Implement Transformer and ViT
- Month 5: Attention visualization tools
- Month 6: Interpretability analysis

**Months 7-9: Wavelet and Hybrids**
- Month 7: Wavelet scattering implementation
- Month 8: Hybrid architectures
- Month 9: Scale importance analysis

**Months 10-12: Comprehensive Comparison**
- Month 10: Standardized evaluation protocol
- Month 11: All comparison studies (data efficiency, robustness)
- Month 12: Statistical analysis, visualization

**Months 13-15: Paper Writing**
- Month 13: Draft Paper 4 (PINN)
- Month 14: Draft Paper 5 (Transformer)
- Month 15: Draft Paper 6 (Comparison)

**Months 16-18: Revisions and Submission**
- Month 16: Internal reviews, revisions
- Month 17: Submit all three papers
- Month 18: Respond to reviews, conference preparation

### 9.2 Resource Requirements

**Computational:**
- GPU: NVIDIA RTX 3090 or better (24 GB VRAM)
- Training time: ~500 GPU hours total
- Storage: 500 GB for models, results

**Personnel:**
- 1 PhD student or postdoc (full-time)
- Advisor time: 20% (guidance, paper review)
- Optional: 1 undergraduate (data processing)

**Software:**
- PyTorch 2.0+
- kymatio (wavelet scattering)
- Standard ML stack (scikit-learn, matplotlib)
- All open-source

**Budget:**
- Cloud compute (if needed): ~$2000
- Conference travel: ~$3000 × 2 = $6000
- Publication fees (if open access): ~$3000 × 3 = $9000
- **Total: ~$17,000**

### 9.3 Risk Mitigation

**Risk 1: PINN doesn't improve performance**
- Mitigation: Ablation studies show contribution of each component
- Fallback: Still publishable as "physics-informed learning in instrumentation"

**Risk 2: Transformers too slow for real-time**
- Mitigation: Optimize with TensorRT, quantization
- Fallback: Position as offline analysis tool, not real-time

**Risk 3: Wavelet accuracy insufficient**
- Mitigation: Combine with learned features (hybrid)
- Fallback: Publish as interpretable alternative for specific applications

**Risk 4: Limited novelty for Paper 6**
- Mitigation: Emphasize systematic comparison, community resource
- Fallback: Target lower-tier but broader journal (JINST, Sensors)

---

## 10. Technical Challenges and Mitigation {#challenges}

### 10.1 Challenge: PINN Physics Loss Design

**Issue:** Differentiable exponential fitting is non-trivial

**Solutions:**
1. **Simplified approach:** Predict τ as auxiliary output, use MSE loss
2. **Pre-computed fitting:** Fit exponentials on CPU, use as targets
3. **Approximation:** Use simple metrics (tail-to-total ratio) as proxy for τ

**Recommended:** Option 1 (multi-task learning)

### 10.2 Challenge: Transformer Computational Cost

**Issue:** Self-attention is O(n²) in sequence length

**Solutions:**
1. **Reduce sequence length:** Downsample or use CNN preprocessing
2. **Efficient attention:** Linformer, Performer (linear attention)
3. **ViT approach:** Patch-based (reduces effective sequence length)

**Recommended:** Option 3 (ViT) + Option 1 (CNN+Trans hybrid)

### 10.3 Challenge: Wavelet Scale Selection

**Issue:** How to choose J and Q parameters?

**Solutions:**
1. **Physics-based:** Match scales to decay times (1/τ)
2. **Grid search:** Try multiple (J, Q) combinations
3. **Adaptive:** Learn scale importance, prune unimportant

**Recommended:** Option 1 informed by Option 2

### 10.4 Challenge: Fair Comparison

**Issue:** Different models have different hyperparameters

**Solutions:**
1. **Standardize:** Same training epochs, early stopping, optimizer
2. **Tune separately:** Grid search for each model
3. **Report both:** "Out-of-box" and "tuned" performance

**Recommended:** Option 2 (fair tuning) + Option 3 (transparency)

### 10.5 Challenge: Interpretability Quantification

**Issue:** How to measure "interpretability"?

**Solutions:**
1. **Qualitative:** Expert assessment, visual inspection
2. **Quantitative:** Correlation with known physics (τ, rise time)
3. **Comparative:** Rank models on interpretability scale

**Recommended:** All three (multi-faceted assessment)

---

## Conclusion

This document outlines a comprehensive research program in advanced machine learning for scintillator classification. The work builds naturally on the baseline study, exploring cutting-edge techniques (PINNs, Transformers, Wavelets) while maintaining scientific rigor.

**Key Strengths:**
- Clear separation from baseline study (future work)
- Well-defined research questions
- Realistic timeline and resources
- High publication impact potential
- Genuine scientific contribution

**Success Criteria:**
- 2-3 papers in top-tier journals
- Open-source code release
- Demonstrated advantages of advanced methods
- Community adoption

**Next Steps:**
1. Complete baseline study (Papers 1-3)
2. Secure funding for advanced ML phase
3. Hire/assign personnel
4. Begin PINN implementation

---

**Document Version:** 1.0  
**Date:** October 2025  
**Status:** Planning / Future Work  
**Prerequisites:** Completion of baseline 3-paper study

---

## Appendices

### Appendix A: Literature Review

**Physics-Informed Neural Networks:**
- Raissi et al. (2019): Original PINN paper
- Karniadakis et al. (2021): Review of PINNs
- Applications in fluid dynamics, materials science

**Transformers in Time Series:**
- Vaswani et al. (2017): Attention is all you need
- Time series transformer applications
- ViT adaptation to 1D signals

**Wavelet Scattering:**
- Mallat (2012): Group invariant scattering
- Andén & Mallat (2014): Deep scattering spectrum
- kymatio library documentation

### Appendix B: Preliminary Results

*(To be filled after initial experiments)*

### Appendix C: Code Repositories

**Planned Releases:**
1. `sipm-pinn`: Physics-informed model implementations
2. `sipm-transformer`: Attention-based models
3. `sipm-wavelet`: Scattering network pipeline
4. `sipm-benchmark`: Comprehensive comparison framework

All under MIT license, hosted on GitHub.

---

**END OF DOCUMENT**
