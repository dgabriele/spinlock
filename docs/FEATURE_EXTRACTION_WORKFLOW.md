# Feature Extraction Workflow for VQTokenizer

This document clarifies the feature extraction process for training VQTokenizer on different datasets.

## Required Features for VQTokenizer Training

The VQTokenizer requires three types of features in the HDF5 dataset:

### 1. **Temporal Features** (`/features/temporal/features`)
- **Shape**: `[N, T, D]` where N=samples, T=timesteps, D=features per timestep
- **What**: Per-timestep time series features (spatial, spectral, cross-channel)
- **How to extract**: Use the CLI command:
  ```bash
  poetry run spinlock extract-features --dataset datasets/your_dataset.h5
  ```
- **Status in QBM**: ✓ Already extracted (247D features, 256 timesteps)

### 2. **Initial Features** (`/features/initial/aggregated/features`)
- **Shape**: `[N, D]` where N=samples, D=initial condition features
- **What**: Statistical/structural features from the initial condition (t=0)
- **How to extract**: Use the script:
  ```bash
  poetry run python scripts/extract_initial_manual_features.py \
      datasets/your_dataset.h5 \
      --statistical  # Use statistical features (recommended)
  ```
- **Status in QBM**: ✓ Extracted (27D statistical features)
- **Note**: This extracts distributional + energy features from the quantum states

### 3. **Parameters** (`/parameters/params`)
- **Shape**: `[N, P]` where N=samples, P=parameter dimensions
- **What**: Raw system parameters (PDE parameters, quantum parameters, etc.)
- **How to extract**: These should already exist in your dataset from generation
- **Status in QBM**: ✓ Present (9D quantum parameters)

### 4. **Raw Input Fields** (`/inputs/fields`)
- **Shape**: `[N, M, C, H, W]` or similar
- **What**: Raw spatial fields at t=0 (used by CNN encoder if enabled)
- **How to extract**: Should already exist from dataset generation
- **Status in QBM**: ✓ Present (3 channels × 2 species × 64×64 spatial)

## Feature Extraction Scripts - What Does What

### Official CLI Commands

1. **`spinlock extract-features`**
   - **Purpose**: Extract TEMPORAL and SUMMARY features
   - **Input**: Raw trajectory dataset with `/outputs/fields` or `/rollouts/`
   - **Output**: Adds `/features/temporal/` and `/features/summary/` to dataset
   - **When to use**: Always run this first after dataset generation

### Standalone Scripts

2. **`scripts/extract_initial_manual_features.py`**
   - **Purpose**: Extract INITIAL features from initial conditions
   - **Input**: Dataset with `/inputs/fields`
   - **Output**: Adds `/features/initial/aggregated/features` to dataset
   - **When to use**: Required for VQTokenizer training
   - **Options**:
     - `--statistical` (default): 27D statistical features (distributional + energy, no spatial)
     - `--pattern`: 42D pattern features (old approach)

3. **`scripts/dev/extract_architecture_initial_features.py`**
   - **Purpose**: Extract ARCHITECTURE features + basic INITIAL features
   - **Input**: Dataset with `/parameters/params` and `/inputs/fields`
   - **Output**: Adds `/features/architecture/` and `/features/initial/`
   - **When to use**: NOT needed for VQTokenizer! This is for a different workflow (legacy)
   - **Note**: Creates simpler initial features (18D) vs the statistical extractor (27D)

4. **`scripts/extract_reference_features.py`**
   - **Purpose**: Extract reference features for MNO-CNO alignment
   - **Input**: MNO dataset + reference CNO dataset
   - **Output**: Copies features from reference dataset to MNO dataset
   - **When to use**: Only for MNO alignment experiments, NOT for standalone VQTokenizer training

5. **`scripts/generate_cno_reference_features.py`**
   - **Purpose**: Generate CNO rollouts for reference feature extraction
   - **When to use**: When building MNO-CNO alignment datasets

## Workflow for Training VQTokenizer on a New Dataset

### Step 1: Generate Dataset
Your dataset should have:
- `/inputs/fields` - Initial conditions
- `/outputs/fields` or `/rollouts/` - Trajectory outputs
- `/parameters/params` - System parameters
- `/metadata/` - Metadata

### Step 2: Extract Temporal Features
```bash
poetry run spinlock extract-features --dataset datasets/your_dataset.h5
```
This adds:
- `/features/temporal/features` [N, T, D]
- `/features/summary/...` (various summary statistics)

### Step 3: Extract Initial Features
```bash
poetry run python scripts/extract_initial_manual_features.py \
    datasets/your_dataset.h5 \
    --statistical
```
This adds:
- `/features/initial/aggregated/features` [N, D]

### Step 4: Verify Features
```bash
poetry run python -c "
import h5py
with h5py.File('datasets/your_dataset.h5', 'r') as f:
    print('Temporal:', f['features/temporal/features'].shape)
    print('Initial:', f['features/initial/aggregated/features'].shape)
    print('Parameters:', f['parameters/params'].shape)
"
```

### Step 5: Create VQTokenizer Config
Adapt `configs/vqvae_50k.yaml` or `configs/vqvae_qbm_50k.yaml`:
- Set `dataset_path` to your dataset
- Update `encoder.initial.manual_dim` to match initial features dimension
- Update `encoder.theta.param_dim` to match parameter dimension
- Update `encoder.initial.in_channels` to match input channels

### Step 6: Train
```bash
poetry run spinlock train-vq-tokenizer --config configs/your_config.yaml
```

## Example: QBM Dataset

Following this workflow for QBM dataset:

1. ✓ Dataset generated: `datasets/qbm_50k.h5`
   - 50K quantum trajectories
   - 9 quantum parameters
   - 3 channels × 2 species × 64×64 spatial

2. ✓ Temporal features extracted: `(50000, 256, 247)`
   - Already present in dataset

3. ✓ Initial features extracted: `(50000, 27)`
   ```bash
   poetry run python scripts/extract_initial_manual_features.py \
       datasets/qbm_50k.h5 --statistical
   ```

4. ✓ Config created: `configs/vqvae_qbm_50k.yaml`
   - `encoder.initial.manual_dim: 27`
   - `encoder.theta.param_dim: 9`
   - `encoder.initial.in_channels: 6`  # 3 channels × 2 species

5. Ready to train:
   ```bash
   poetry run spinlock train-vq-tokenizer --config configs/vqvae_qbm_50k.yaml
   ```

## Common Confusion Points

### Q: What's the difference between "initial" and "architecture" features?
**A**:
- **Initial features**: Derived from spatial initial conditions at t=0 (e.g., density distributions, gradients)
- **Architecture features**: Derived from raw parameters themselves (parameter products, ratios, etc.)
- VQTokenizer uses **initial features** + raw parameters (not architecture features)

### Q: Why are there multiple feature extraction scripts?
**A**: Historical reasons. The main workflow is:
1. `spinlock extract-features` (CLI) → temporal/summary
2. `scripts/extract_initial_manual_features.py` → initial

Other scripts are for specialized workflows (MNO alignment, legacy approaches).

### Q: Do I need CNN pretrained weights?
**A**:
- PDE datasets: Yes, use `checkpoints/v2/cnn_pretrained.pt`
- QBM datasets: No, train from scratch (set `pretrained_cnn_path: null`)
- The CNN encoder processes raw spatial fields in addition to manual features

### Q: What if my dataset has different structure?
**A**: The feature extractors expect:
- Inputs: `[N, M, C, H, W]` or `[N, C, H, W]` spatial fields
- Outputs: `[N, M, T, C, H, W]` or similar trajectory structure
If your structure is different, you may need to adapt the extraction scripts.

