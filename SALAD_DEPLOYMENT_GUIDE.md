# Salad Cloud QBM Dataset Generation - Deployment Guide

## Quick Start: Small-Scale Test (100 samples)

### Prerequisites

1. **Environment Variables** (set these before running):
```bash
export SALAD_API_KEY="your-salad-api-key"
export DOCKER_USERNAME="psilogon"
export DOCKER_PASSWORD="your-docker-password"
export MINIO_ACCESS_KEY="your-minio-access-key"
export MINIO_SECRET_KEY="your-minio-secret-key"
```

2. **MinIO/Cloudflare Tunnel Running**:
```bash
# In separate terminal:
cloudflare-tunnel-to-minio  # Or however you start it
# Should expose: https://calculator-cargo-intend-mixer.trycloudflare.com
```

3. **MinIO Bucket Created**:
```bash
aws s3 mb s3://spinlock-datasets \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com
```

---

## Step 1: Build and Push Updated Docker Image

The Docker image was updated with:
- Latest code (Sobol offset, generation launcher, etc.)
- AWS CLI for S3 uploads
- Updated entrypoint for generation jobs

```bash
cd /home/daniel/projects/spinlock

# Build base image (if not already built recently)
docker build -f docker/base/Dockerfile -t spinlock:base .

# Build Salad image with generation support
docker build -f docker/training/Dockerfile.salad -t psilogon/spinlock:latest .

# Push to Docker Hub
docker push psilogon/spinlock:latest
```

**Expected time**: ~5-10 minutes (base layer is cached if recently built)

---

## Step 2: Launch Small-Scale Test (100 samples, 1 job)

```bash
poetry run spinlock launch-salad-generation \
    --config configs/distributed/salad_qbm_generation_test.yaml \
    --verbose
```

**What this does**:
- Creates 1 Salad container with RTX 4060 Ti
- Generates 100 QBM samples with wide parameter ranges
- Uploads result to `s3://spinlock-datasets/qbm_test/part00.h5`
- GPU-accelerated: ~2-3 minutes total
- Cost: ~$0.01

**Expected output**:
```
============================================================
SALAD CLOUD DATASET GENERATION
============================================================
[SaladJobLauncher] Initialized for job spinlock-a1b2c3d4
[SaladGenerationLauncher] Splitting 100 samples into 1 jobs of 100 samples each
  Job 0: offset=0, samples=100, output=s3://spinlock-datasets/qbm_test/part00.h5
[SaladJobLauncher] Built 1 container specs
[SaladJobLauncher] Creating container group 1/1: spinlock-a1b2c3d4-job-0...
  ✓ Container group created: spinlock-a1b2c3d4-job-0
[SaladGenerationLauncher] Monitoring generation jobs...
  Waiting for 1 jobs to complete (polling every 30s)...
  ✓ Job 0 completed (1/1) [120s elapsed]
  ✓ All jobs completed in 120s (2.0m)
[SaladGenerationLauncher] ✓ All generation jobs completed
```

---

## Step 3: Monitor Progress

### Option 1: Salad Dashboard
- Go to https://portal.salad.com
- Navigate to your project: `spinlock-datasets`
- Check container group status and logs

### Option 2: CLI Logs
The launcher polls status every 30s and prints updates. Wait for:
```
✓ Job 0 completed (1/1)
```

---

## Step 4: Download and Verify Result

```bash
# Download from MinIO
aws s3 cp s3://spinlock-datasets/qbm_test/part00.h5 /tmp/test_result.h5 \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com

# Verify dataset
poetry run python -c "
import h5py
import numpy as np

with h5py.File('/tmp/test_result.h5', 'r') as f:
    print('✓ File opened successfully')
    print(f'  Samples: {f[\"/parameters/params\"].shape[0]}')

    # Check parameter ranges (should be 3× wider)
    params = f['/parameters/params'][:]
    print(f'  Parameter ranges:')
    for dim in range(params.shape[1]):
        print(f'    Dim {dim}: [{params[:, dim].min():.6f}, {params[:, dim].max():.6f}]')

    # Check features
    if '/features/temporal/per_timestep' in f:
        print(f'  Features shape: {f[\"/features/temporal/per_timestep\"].shape}')

    print('✓ Dataset valid')
"
```

**Expected output**:
```
✓ File opened successfully
  Samples: 100
  Parameter ranges:
    Dim 0: [0.000045, 0.287123]  # gamma (wider)
    Dim 1: [0.003124, 28.456789]  # kT (wider)
    ...
  Features shape: (100, 256, 314)  # 100 samples × 256 timesteps × 314 features
✓ Dataset valid
```

---

## Step 5: Full-Scale Deployment (1M samples)

Once the test succeeds, deploy full-scale:

```bash
poetry run spinlock launch-salad-generation \
    --config configs/distributed/salad_qbm_generation.yaml \
    --verbose
```

**Full-scale parameters**:
- 10 jobs × 100K samples each = 1M total
- GPU-accelerated: ~2-3 hours total (jobs run in parallel)
- Cost: ~$2-3 total
- Output: 10 files in `s3://spinlock-datasets/qbm_1m_wide/part*.h5`

---

## Step 6: Merge Results (Full-Scale Only)

After all 10 jobs complete:

```bash
# Download all parts
aws s3 sync s3://spinlock-datasets/qbm_1m_wide/ datasets/parts/ \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com

# Merge into single file
poetry run spinlock merge-datasets \
    --input "datasets/parts/qbm_1m_wide_part*.h5" \
    --output datasets/qbm_1m_wide.h5 \
    --validate \
    --verbose
```

**Expected output**:
```
[MergeDatasets] Found 10 files to merge
[MergeDatasets] Merging into datasets/qbm_1m_wide.h5...
  ✓ Merged /parameters/params: 1,000,000 total samples
  ✓ Merged /features/temporal/per_timestep: 1,000,000 total samples
[MergeDatasets] Validating merged dataset...
  Checking 1,000,000 parameter sets for duplicates...
  ✓ No duplicate parameter sets found
[MergeDatasets] ✓ Validation complete
✓ MERGE COMPLETE
Total samples: 1,000,000
File size: 147.32 GB
```

---

## Troubleshooting

### Container fails to start
**Check**: Docker image was pushed successfully
```bash
docker pull psilogon/spinlock:latest
```

### S3 upload fails
**Check**: MinIO credentials and endpoint URL
```bash
# Test credentials
aws s3 ls s3://spinlock-datasets/ \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com
```

### Job timeout or stuck
**Check**: Salad dashboard logs for errors
- Common issues: GPU allocation failed, out of memory
- Solution: Try different GPU class or reduce batch size

### Merge fails with duplicates
**Check**: Sobol offsets were applied correctly
- Each job should have unique offset: 0, 100000, 200000, ...
- If duplicates found, re-run failed jobs with correct offset

---

## Cost Estimate

### Test (100 samples, 1 job)
- 1× RTX 4060 Ti × 2-3 mins @ ~$0.07/hr
- **Total: ~$0.01**

### Full-scale (1M samples, 10 jobs)
- 10× RTX 4060 Ti × 20-30 mins @ ~$0.07/hr
- **Total: ~$2-3**

---

## Next Steps After Generation

1. **Train VQTokenizer** on 1M dataset:
```bash
poetry run spinlock train-vq-tokenizer \
    --config configs/qbm/vqvae_diverse_v2.yaml \
    --dataset datasets/qbm_1m_wide.h5 \
    --output checkpoints/qbm/vqvae_1m_wide/
```

2. **Measure diversity improvement**:
- Expected: 45-60% unique token patterns (vs 17.1% baseline)
- Zero-variance features: <20 (vs 78 baseline)

3. **Train D3PM** on improved tokens
4. **Bootstrap LLM alignment** with diverse token space
