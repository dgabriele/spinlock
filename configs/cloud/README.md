# Cloud Dataset Generation Configs

This directory contains configurations for generating datasets on cloud GPU providers.

## Quick Start

### 1. Setup API Key

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit and add your Lambda Labs API key
# Get it from: https://cloud.lambdalabs.com/api-keys
nano .env
```

Your `.env` should look like:
```bash
LAMBDA_API_KEY=secret_noa_your-actual-key-here
```

### 2. Test with Minimal Dataset (10 operators)

```bash
# Verify API key is loaded
source .env && echo "API key: ${LAMBDA_API_KEY:0:20}..."

# Run minimal test (10 operators, ~2 minutes, ~$0.02)
env LAMBDA_API_KEY=$LAMBDA_API_KEY poetry run spinlock cloud-generate \
    --config configs/cloud/lambda_labs_test_10.yaml \
    --provider lambda_labs \
    --monitor
```

### 3. Run Production 100K Dataset

```bash
# Generate 100K operators (~25-35 hours, ~$30-40)
env LAMBDA_API_KEY=$LAMBDA_API_KEY poetry run spinlock cloud-generate \
    --config configs/cloud/lambda_labs_100k.yaml \
    --provider lambda_labs \
    --monitor
```

## Available Configs

### `lambda_labs_test_10.yaml`
- **Purpose**: Infrastructure testing and validation
- **Scale**: 10 operators × 5 realizations = 50 samples
- **Runtime**: 1-2 minutes
- **Cost**: ~$0.02
- **Storage**: Lambda Labs built-in file storage (no S3)
- **Output**: `datasets/lambda_labs_test_10.h5` (~10MB)

### `lambda_labs_100k.yaml`
- **Purpose**: Production VQ-VAE training dataset
- **Scale**: 100,000 operators × 5 realizations = 500,000 samples
- **Runtime**: 17-52 hours (estimated 25-35 hours)
- **Cost**: $19-58 (estimated $30-40)
- **Storage**: S3 (configurable to Lambda Labs file storage)
- **Output**: `s3://spinlock-datasets/production/100k/baseline_100k.h5` (~10GB)

## Storage Options

### Lambda Labs Built-in File Storage (Default for Testing)

Stores datasets directly on Lambda Labs instance, then downloads via SCP.

**Pros**:
- No S3 dependency
- Simple setup (no AWS credentials needed)
- Free (no storage costs)

**Cons**:
- Manual download required
- Limited to instance disk space

**Config**:
```yaml
cloud:
  lambda_labs_storage:
    remote_path: "/home/ubuntu/datasets"
    local_download_path: "datasets/"
```

### S3 Storage (Recommended for Production)

Automatically uploads dataset to S3 after generation.

**Pros**:
- Automatic upload
- Durable storage
- Easy sharing
- Integrated with AWS ecosystem

**Cons**:
- Requires AWS credentials
- Storage costs (~$0.23/month for 10GB)

**Config**:
```yaml
cloud:
  s3:
    bucket: "spinlock-datasets"
    prefix: "production/100k/"
    region: "us-west-2"
    storage_class: "STANDARD"  # or INTELLIGENT_TIERING
```

## Feature Extraction

Both configs automatically extract INITIAL and TEMPORAL features:

### INITIAL Features
- IC type metadata (`gaussian_random_field_v0-4`, `multiscale_grf_*`, `structured`, `localized`)
- Initial condition statistics (variance, correlation length, etc.)
- Stored in HDF5 `ic_types` field

### TEMPORAL Features
- **Per-timestep** (46 features): spatial, spectral, cross-channel
- **Per-trajectory** (100+ features): temporal, causality, invariant_drift
- **Operator sensitivity** (12 features): extracted inline during generation
- **Total**: ~146 aggregated features per operator
- Stored in HDF5 `features/summary/` group

**Feature-only mode**: Raw trajectories NOT stored (saves 120TB → <10GB)

## Cost Estimates

### Lambda Labs A100 Pricing
- **Hourly rate**: $1.10/hr
- **GPU**: NVIDIA A100 (40GB VRAM)
- **Region**: us-west-1

### Test Run (10 operators)
- Runtime: 1-2 minutes
- Cost: ~$0.02
- Purpose: Validate infrastructure

### Production Run (100K operators)
- Runtime: 17-52 hours
- Expected: 25-35 hours
- Cost: $19-58
- Expected: $30-40

### S3 Storage (optional)
- Storage: ~$0.023/GB/month × 10GB = $0.23/month
- PUT requests: $0.005 per 1000 × ~100 = $0.50
- Total one-time: ~$0.50

## Safety Limits

Both configs include safety limits:

```yaml
cloud:
  lambda_labs:
    max_cost_per_hour: 1.50  # Will refuse if instance costs more
    auto_shutdown_on_completion: true  # Terminate after job completes
```

## Monitoring

Monitor job progress:

```bash
# Get job ID from submission
JOB_ID=abc12345

# Monitor progress
poetry run spinlock cloud-generate \
    --provider lambda_labs \
    --job-id $JOB_ID \
    --monitor
```

## Troubleshooting

### API Key Not Found
```bash
# Verify .env file exists
cat .env | grep LAMBDA_API_KEY

# Source it before running
source .env
env | grep LAMBDA_API_KEY
```

### Config Validation Errors
```bash
# Check config syntax
poetry run python -c "
from spinlock.config import load_config
from pathlib import Path
config = load_config(Path('configs/cloud/lambda_labs_test_10.yaml'))
print('✓ Config valid')
"
```

### SSH Key Issues
1. Generate SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/lambda_labs_key`
2. Add to Lambda Labs: https://cloud.lambdalabs.com/ssh-keys
3. Update config: `ssh_key_name: "spinlock-key"`

## Next Steps

After dataset generation:

1. **Download dataset** (if using Lambda Labs file storage):
   ```bash
   ls -lh datasets/lambda_labs_test_10.h5
   ```

2. **Verify features** were extracted:
   ```bash
   h5ls -r datasets/lambda_labs_test_10.h5
   ```

3. **Train VQ-VAE**:
   ```bash
   poetry run spinlock train-vqvae \
       --dataset datasets/lambda_labs_test_10.h5 \
       --config configs/vqvae/baseline.yaml
   ```

## Support

- Lambda Labs API: https://cloud.lambdalabs.com/api-keys
- Lambda Labs Docs: https://docs.lambdalabs.com/
- Spinlock Issues: https://github.com/your-org/spinlock/issues
