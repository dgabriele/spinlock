# VQ-Led Loss Feature Dimension Mismatch

## Problem

L_vq-driven NOA training fails with dimension mismatch error:

```
RuntimeError: The size of tensor a (171) must match the size of tensor b (187) at non-singleton dimension 1
```

## Root Cause

1. **VQ-VAE Training**: VQ-VAE was trained on **171 cleaned features** (after feature masking)
   - Original features: 270
   - After cleaning mask: 143
   - VQ-VAE input_dim: 171 (from checkpoint `model_config`)

2. **Feature Extraction**: NOA feature extractor produces **187 raw features**
   - This includes all features before cleaning

3. **VQ-Led Loss Bug**: The loss tries to compare:
   - `recon_features` = VQ-VAE decode output → [B, 171]
   - `features_norm` = normalized extracted features → [B, 187]

## Location

`src/spinlock/noa/losses/vq_led.py:167`

```python
recon_features = self.alignment.vqvae.decode(z_q_list)  # [B, 171]
recon_loss = F.mse_loss(recon_features, features_norm)  # [B, 187] - MISMATCH!
```

## Solution Options

### Option 1: Apply Feature Cleaning in VQ-Led Loss (Recommended)

Store and apply the same feature cleaning/masking used during VQ-VAE training:

```python
# In VQVAEAlignmentLoss.__init__():
self.feature_mask = checkpoint.get('feature_mask')
self.feature_cleaning_params = checkpoint.get('feature_cleaning_params')

# In VQLedLoss.forward():
features_norm = self.alignment._normalize_features(pred_features)

# Apply cleaning to match VQ-VAE input
if self.alignment.feature_mask is not None:
    features_cleaned = features_norm[:, self.alignment.feature_mask]
else:
    features_cleaned = features_norm

# Now dimensions match
recon_features = self.alignment.vqvae.decode(z_q_list)  # [B, 171]
recon_loss = F.mse_loss(recon_features, features_cleaned)  # [B, 171] - OK!
```

### Option 2: Adjust VQ-VAE to Handle Raw Features

Modify VQ-VAE to internally apply feature masking, so it can accept 187-dim input and only use 171 dims.

### Option 3: Adjust Feature Extractor

Make feature extractor output only the 171 cleaned features - but this breaks existing MSE-led training which may expect all features.

## Impact

- L_vq-driven training completely broken (skips all batches)
- L_traj-driven training unaffected (doesn't use VQ-VAE reconstruction loss)

## Status

- ✅ Root cause identified
- ❌ Fix not yet implemented
- Workaround: Use L_traj-driven training only

## Related Files

- `src/spinlock/noa/losses/vq_led.py` - VQ-led loss implementation
- `src/spinlock/noa/vqvae_alignment.py` - VQ-VAE alignment and loading
- `checkpoints/production/100k_3family_v1/` - VQ-VAE checkpoint with feature_mask
