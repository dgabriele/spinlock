# Parameter Sensitivity Architecture Design

**Design Principles**: DRY, OOP, functional decomposition, reusability

---

## Component Hierarchy

```
BaseNOALoss (existing)
    │
    ├─ MSELedLoss (existing)
    │
    └─ ParameterSensitiveLoss (NEW)
           │
           ├─ Uses: ParameterReconstructor
           ├─ Uses: ContrastiveSimilarity
           └─ Uses: SensitivityMetric
```

---

## 1. Feature Extractors (Reusable Components)

### 1.1 RolloutFeatureExtractor

**Purpose**: Extract summary features from rollouts (DRY - used by multiple components)
**File**: `src/spinlock/mno/features/rollout_features.py` (NEW)

```python
class RolloutFeatureExtractor(nn.Module):
    """Extract statistical summary features from rollouts.

    Reusable component for:
    - Parameter reconstruction
    - Contrastive learning
    - Sensitivity metrics

    Features extracted:
    - Temporal statistics: mean, std, min, max over time
    - Spatial statistics: mean, std, min, max over space
    - Spectral features: FFT magnitudes at key frequencies
    - Total: 32D feature vector
    """

    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, rollout: Tensor) -> Tensor:
        """
        Args:
            rollout: [B, T, C, H, W]
        Returns:
            features: [B, feature_dim]
        """
        # Temporal stats [B, 4]
        temporal = torch.stack([
            rollout.mean(dim=1).flatten(1).mean(1),  # mean over time
            rollout.std(dim=1).flatten(1).mean(1),   # std over time
            rollout.min(dim=1)[0].flatten(1).mean(1),
            rollout.max(dim=1)[0].flatten(1).mean(1),
        ], dim=1)

        # Spatial stats [B, 4]
        spatial = torch.stack([
            rollout.mean(dim=(3,4)).flatten(1).mean(1),  # mean over space
            rollout.std(dim=(3,4)).flatten(1).mean(1),
            # ... etc
        ], dim=1)

        # Spectral features [B, 8] - FFT of temporal signal
        # ... (extract dominant frequencies)

        # Concatenate all features [B, 32]
        features = torch.cat([temporal, spatial, spectral], dim=1)
        return features
```

---

## 2. Parameter Reconstruction (Extends existing pattern)

### 2.1 ParameterReconstructor

**Purpose**: Predict parameters from rollout features
**File**: `src/spinlock/mno/modules/parameter_reconstructor.py` (NEW)

```python
class ParameterReconstructor(nn.Module):
    """Reconstruct parameters from rollout features.

    Uses RolloutFeatureExtractor (DRY) + MLP predictor.
    Follows same pattern as existing modules.
    """

    def __init__(
        self,
        param_dim: int = 14,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.param_dim = param_dim

        # Reuse feature extractor (DRY)
        self.feature_extractor = RolloutFeatureExtractor(feature_dim)

        # MLP predictor
        layers = []
        in_dim = feature_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, param_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, rollout: Tensor) -> Tensor:
        """
        Args:
            rollout: [B, T, C, H, W]
        Returns:
            params_pred: [B, param_dim]
        """
        features = self.feature_extractor(rollout)
        params_pred = self.mlp(features)
        return params_pred
```

### 2.2 ParameterReconstructionLoss

**Purpose**: Modular loss component
**File**: `src/spinlock/mno/losses/components/parameter_reconstruction.py` (NEW)

```python
class ParameterReconstructionLoss(nn.Module):
    """Compute parameter reconstruction loss.

    Follows BaseNOALoss pattern but as a modular component.
    Can be used standalone or composed into ParameterSensitiveLoss.
    """

    def __init__(
        self,
        param_reconstructor: Optional[ParameterReconstructor] = None,
        param_dim: int = 14,
    ):
        super().__init__()
        self.reconstructor = param_reconstructor or ParameterReconstructor(param_dim)

    def forward(
        self,
        rollout: Tensor,
        params_true: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            rollout: [B, T, C, H, W]
            params_true: [B, param_dim]
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        params_pred = self.reconstructor(rollout)
        loss = F.mse_loss(params_pred, params_true)

        # Accuracy metric: % of params within 10% of true value
        relative_error = torch.abs((params_pred - params_true) / (params_true + 1e-6))
        accuracy = (relative_error < 0.1).float().mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'params_pred': params_pred,  # For debugging
        }
```

---

## 3. Contrastive Learning (Reusable similarity metric)

### 3.1 ContrastiveSimilarity

**Purpose**: Compute similarity between rollouts and parameters
**File**: `src/spinlock/mno/modules/contrastive_similarity.py` (NEW)

```python
class ContrastiveSimilarity(nn.Module):
    """Compute contrastive similarity between rollouts and parameters.

    Projects both to shared embedding space for similarity computation.
    Reusable for InfoNCE, triplet loss, etc.
    """

    def __init__(
        self,
        param_dim: int = 14,
        rollout_feature_dim: int = 32,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # Reuse feature extractor (DRY)
        self.rollout_extractor = RolloutFeatureExtractor(rollout_feature_dim)

        # Project to shared embedding space
        self.rollout_proj = nn.Sequential(
            nn.Linear(rollout_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.param_proj = nn.Sequential(
            nn.Linear(param_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        rollout: Tensor,
        params: Tensor,
    ) -> Tensor:
        """
        Args:
            rollout: [B, T, C, H, W]
            params: [B, param_dim]
        Returns:
            similarity_matrix: [B, B] cosine similarities
        """
        # Extract and project
        rollout_features = self.rollout_extractor(rollout)  # [B, 32]
        rollout_emb = self.rollout_proj(rollout_features)   # [B, 64]
        param_emb = self.param_proj(params)                 # [B, 64]

        # L2 normalize
        rollout_emb = F.normalize(rollout_emb, p=2, dim=1)
        param_emb = F.normalize(param_emb, p=2, dim=1)

        # Similarity matrix
        similarity = rollout_emb @ param_emb.T  # [B, B]
        return similarity
```

### 3.2 ContrastiveLoss

**Purpose**: InfoNCE-style contrastive loss
**File**: `src/spinlock/mno/losses/components/contrastive.py` (NEW)

```python
class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for rollout-parameter alignment.

    Uses ContrastiveSimilarity (DRY) for embedding computation.
    """

    def __init__(
        self,
        similarity_module: Optional[ContrastiveSimilarity] = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.similarity = similarity_module or ContrastiveSimilarity()
        self.temperature = temperature

    def forward(
        self,
        rollout: Tensor,
        params: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            rollout: [B, T, C, H, W]
            params: [B, param_dim]
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        B = rollout.shape[0]

        # Compute similarities
        logits = self.similarity(rollout, params) / self.temperature  # [B, B]

        # Diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Accuracy: % of samples where positive has highest similarity
        pred_labels = logits.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
        }
```

---

## 4. Sensitivity Metric (Functional, stateless)

### 4.1 SensitivityMetric

**Purpose**: Measure parameter sensitivity via finite differences
**File**: `src/spinlock/mno/metrics/sensitivity.py` (NEW)

```python
def compute_parameter_sensitivity(
    mno: nn.Module,
    ic: Tensor,
    params: Tensor,
    epsilon: float = 0.01,
    num_samples: int = 5,
) -> Dict[str, float]:
    """Compute parameter sensitivity metric.

    Functional (stateless) - can be called from anywhere.
    Used by ParameterSensitivityLoss and validation.

    Args:
        mno: MNO model
        ic: Initial condition [B, C, H, W]
        params: Parameters [B, param_dim]
        epsilon: Perturbation magnitude
        num_samples: Number of random perturbations to average

    Returns:
        Dict with sensitivity metrics
    """
    with torch.no_grad():
        # Baseline rollout
        rollout_0 = mno.rollout(ic, params=params, steps=256, return_all_steps=True)

        sensitivities = []
        for _ in range(num_samples):
            # Random perturbation
            delta = epsilon * torch.randn_like(params)
            params_perturbed = params + delta

            # Perturbed rollout
            rollout_1 = mno.rollout(ic, params=params_perturbed, steps=256, return_all_steps=True)

            # Measure change
            diff = (rollout_1 - rollout_0).abs().mean()
            sensitivity = diff / epsilon
            sensitivities.append(sensitivity.item())

        # Compute temporal variance for comparison
        temporal_var = rollout_0.var(dim=1).mean().item()

        return {
            'sensitivity_mean': np.mean(sensitivities),
            'sensitivity_std': np.std(sensitivities),
            'temporal_variance': temporal_var,
            'diversity_ratio': np.mean(sensitivities) / temporal_var if temporal_var > 0 else 0.0,
        }
```

### 4.2 ParameterSensitivityLoss

**Purpose**: Loss component for sensitivity regularization
**File**: `src/spinlock/mno/losses/components/sensitivity.py` (NEW)

```python
class ParameterSensitivityLoss(nn.Module):
    """Regularization loss for parameter sensitivity.

    Penalizes if parameter changes don't affect outputs enough.
    Uses compute_parameter_sensitivity for measurement (DRY).
    """

    def __init__(
        self,
        target_ratio: float = 0.1,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.epsilon = epsilon

    def forward(
        self,
        mno: nn.Module,
        ic: Tensor,
        params: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            mno: MNO model (needs to be passed in)
            ic: Initial condition [B, C, H, W]
            params: Parameters [B, param_dim]
        Returns:
            Dict with 'loss' and metrics
        """
        # Compute sensitivity (uses shared function - DRY)
        metrics = compute_parameter_sensitivity(
            mno, ic, params,
            epsilon=self.epsilon,
            num_samples=1,  # Single sample for efficiency during training
        )

        # Loss: penalize deviation from target ratio
        current_ratio = metrics['diversity_ratio']
        target = self.target_ratio
        loss = F.mse_loss(
            torch.tensor(current_ratio, device=ic.device),
            torch.tensor(target, device=ic.device),
        )

        return {
            'loss': loss,
            **metrics,  # Include metrics for logging
        }
```

---

## 5. Composed Loss (Clean composition)

### 5.1 ParameterSensitiveLoss

**Purpose**: Composes all components following BaseNOALoss pattern
**File**: `src/spinlock/mno/losses/parameter_sensitive.py` (NEW)

```python
class ParameterSensitiveLoss(BaseNOALoss):
    """Parameter-sensitive MNO training loss.

    Composes:
    - MSE trajectory loss (existing pattern)
    - Parameter reconstruction loss (modular component)
    - Contrastive loss (modular component)
    - Sensitivity regularization (modular component)

    Follows BaseNOALoss interface for compatibility with existing training loop.
    """

    def __init__(
        self,
        lambda_traj: float = 1.0,
        lambda_ic: float = 0.3,
        lambda_param_recon: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_sensitivity: float = 0.2,
        param_reconstructor: Optional[ParameterReconstructor] = None,
        contrastive_similarity: Optional[ContrastiveSimilarity] = None,
    ):
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_ic = lambda_ic
        self.lambda_param_recon = lambda_param_recon
        self.lambda_contrastive = lambda_contrastive
        self.lambda_sensitivity = lambda_sensitivity

        # Modular loss components (can be swapped/configured)
        self.param_recon_loss = ParameterReconstructionLoss(param_reconstructor)
        self.contrastive_loss = ContrastiveLoss(contrastive_similarity)
        self.sensitivity_loss = ParameterSensitivityLoss()

    def compute(
        self,
        pred_trajectory: Tensor,
        target_trajectory: Tensor,
        ic: Optional[Tensor] = None,
        mno: Optional[nn.Module] = None,
        params: Optional[Tensor] = None,  # NEW: need params for sensitivity losses
    ) -> LossOutput:
        """Compute all loss components.

        Follows BaseNOALoss.compute() signature but adds params argument.
        """
        components = {}

        # 1. Standard trajectory loss (existing pattern)
        L_traj = F.mse_loss(pred_trajectory, target_trajectory)
        components['traj'] = L_traj

        # 2. IC reconstruction (existing pattern)
        if ic is not None and self.lambda_ic > 0:
            pred_ic = pred_trajectory[:, 0]
            L_ic = F.mse_loss(pred_ic, ic)
            components['ic'] = L_ic

        # 3. Parameter reconstruction (NEW - modular)
        if self.lambda_param_recon > 0 and params is not None:
            param_recon_out = self.param_recon_loss(pred_trajectory, params)
            components['param_recon'] = param_recon_out['loss']
            components['param_recon_acc'] = param_recon_out['accuracy']

        # 4. Contrastive loss (NEW - modular)
        if self.lambda_contrastive > 0 and params is not None:
            contrastive_out = self.contrastive_loss(pred_trajectory, params)
            components['contrastive'] = contrastive_out['loss']
            components['contrastive_acc'] = contrastive_out['accuracy']

        # 5. Sensitivity regularization (NEW - modular)
        if self.lambda_sensitivity > 0 and mno is not None and params is not None:
            sensitivity_out = self.sensitivity_loss(mno, ic, params)
            components['sensitivity'] = sensitivity_out['loss']
            components['diversity_ratio'] = sensitivity_out['diversity_ratio']

        # Weighted sum
        total = (
            self.lambda_traj * components.get('traj', 0) +
            self.lambda_ic * components.get('ic', 0) +
            self.lambda_param_recon * components.get('param_recon', 0) +
            self.lambda_contrastive * components.get('contrastive', 0) +
            self.lambda_sensitivity * components.get('sensitivity', 0)
        )

        # Convert to metrics for logging
        metrics = {k: v.item() if isinstance(v, Tensor) else v
                   for k, v in components.items()}

        return LossOutput(
            total=total,
            components=components,
            metrics=metrics,
        )

    @property
    def leading_loss_name(self) -> str:
        return "traj"  # Still physics-led

    @property
    def auxiliary_loss_names(self) -> List[str]:
        return ["ic", "param_recon", "contrastive", "sensitivity"]
```

---

## 6. Training Integration (Minimal changes to existing code)

### 6.1 Update Training Loop

**File**: `src/spinlock/cli/train_meta_operator.py` (MODIFY)

```python
# In training loop - only need to pass params to loss.compute()
loss_output = loss_fn.compute(
    pred_trajectory=pred_traj,
    target_trajectory=target_traj,
    ic=ic,
    mno=model,  # Pass model reference
    params=params,  # NEW: pass params
)
```

### 6.2 FiLM Learning Rate Scaling

**File**: `src/spinlock/cli/train_meta_operator.py` (MODIFY)

```python
# Separate param groups for differential learning rates
def create_optimizer(model, base_lr, film_lr_multiplier=1.0):
    """Create optimizer with separate LR for FiLM parameters."""
    film_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'film' in name.lower():
            film_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': base_lr},
        {'params': film_params, 'lr': base_lr * film_lr_multiplier},
    ]

    return torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
```

---

## Summary

**DRY Principles Applied:**
- ✓ `RolloutFeatureExtractor` used by param_recon, contrastive, and sensitivity
- ✓ `compute_parameter_sensitivity` function shared by loss and validation
- ✓ Modular loss components that can be reused independently

**OOP Patterns:**
- ✓ Follows existing `BaseNOALoss` interface
- ✓ Composition over inheritance (uses loss components)
- ✓ Clear separation of concerns (feature extraction, similarity, loss computation)

**Functional Decomposition:**
- ✓ Small, focused modules (<100 lines each)
- ✓ Single responsibility principle
- ✓ Testable components

**Reusability:**
- ✓ Each component can be used standalone or composed
- ✓ Easy to swap implementations (e.g., different feature extractors)
- ✓ Minimal changes to existing training loop

This architecture is maintainable, extensible, and follows senior ML engineering best practices.
