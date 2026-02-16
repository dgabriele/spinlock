# Salad Cloud QBM Generation - Resume Point

**Date**: 2026-02-16
**Status**: Implementation complete, blocked on Salad org/project configuration

---

## Current State: Ready to Deploy (99% Complete)

### ✅ What's Done

1. **Code Implementation** (100% complete)
   - ✅ Sobol offset support in sampler (`src/spinlock/sampling/sobol.py`)
   - ✅ CLI `--sobol-offset` argument (`src/spinlock/cli/generate.py`)
   - ✅ Config schema updated (`src/spinlock/config/schema.py`)
   - ✅ Pipeline integration (`src/spinlock/dataset/pipeline.py`)
   - ✅ Launcher base class extracted (`src/spinlock/distributed/salad/base_launcher.py`)
   - ✅ Training launcher separated (`src/spinlock/distributed/salad/training_launcher.py`)
   - ✅ Generation launcher created (`src/spinlock/distributed/salad/generation_launcher.py`)
   - ✅ CLI commands registered (`launch-salad-generation`, `merge-datasets`)
   - ✅ All tests pass (`scripts/test_sobol_offset.py`)

2. **Docker Infrastructure** (100% complete)
   - ✅ Base image built: `spinlock:base`
   - ✅ Salad image built: `psilogon/spinlock:latest`
   - ✅ AWS CLI installed for S3 uploads
   - ✅ Updated entrypoint for generation jobs (`docker/training/entrypoint-salad.sh`)
   - ✅ Pushed to Docker Hub successfully

3. **Configuration Files** (100% complete)
   - ✅ Wide parameter ranges: `configs/experiments/qbm_1m_wide.yaml` (3× expansion)
   - ✅ Test config: `configs/distributed/salad_qbm_generation_test.yaml` (100 samples)
   - ✅ Full config: `configs/distributed/salad_qbm_generation.yaml` (1M samples)
   - ✅ Batch size optimized for Salad GPUs: 200 (vs 50 local)

4. **Documentation** (100% complete)
   - ✅ `SALAD_DEPLOYMENT_GUIDE.md` - Step-by-step deployment instructions
   - ✅ `IMPLEMENTATION_SUMMARY.md` - Technical overview
   - ✅ `scripts/test_sobol_offset.py` - Validation tests (all passing)

### 🔴 Blocked On: Salad Organization/Project Names

**Error**: `401 Unauthorized` when creating container groups

**What we tried**:
- Organization: `spinlock`, Project: `spinlock-datasets` → 401
- Organization: `spinlock`, Project: `spinlock` → 401

**What we know**:
- User has a Salad project called "spinlock"
- API key is valid and loaded correctly
- SDK connects successfully
- Need correct organization name from Salad dashboard

**Required information**:
1. **Organization name** from https://portal.salad.com URL
   - Look for: `https://portal.salad.com/organizations/YOUR-ORG-NAME/...`
2. **Project name** - confirm exact spelling (user said "spinlock")

---

## How to Resume

### Step 1: Get Correct Org/Project Names

Visit https://portal.salad.com and:
1. Note the organization name from the URL
2. Navigate to your "spinlock" project
3. Confirm exact project name (case-sensitive)

### Step 2: Update Config File

Edit `configs/distributed/salad_qbm_generation_test.yaml`:

```yaml
salad:
  organization: "YOUR-ACTUAL-ORG-NAME"  # ← Update this
  project: "spinlock"                   # ← Confirm this is exact
```

### Step 3: Launch Test Deployment

```bash
cd /home/daniel/projects/spinlock

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Launch 100-sample test (~2-3 mins, ~$0.01)
poetry run spinlock launch-salad-generation \
    --config configs/distributed/salad_qbm_generation_test.yaml \
    --verbose
```

### Step 4: Monitor Progress

The launcher polls every 30s and will show:
```
✓ Job 0 completed (1/1) [120s elapsed]
✓ All jobs completed
```

### Step 5: Download and Verify

```bash
# Download result
aws s3 cp s3://spinlock-datasets/qbm_test/qbm_test_part00.h5 /tmp/test_result.h5 \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com

# Verify dataset
poetry run python -c "
import h5py
with h5py.File('/tmp/test_result.h5', 'r') as f:
    print(f'Samples: {f[\"/parameters/params\"].shape[0]}')
    print(f'Expected: 100')
"
```

### Step 6: Full Deployment (After Test Succeeds)

Update `configs/distributed/salad_qbm_generation.yaml` with correct org/project, then:

```bash
# Launch 1M sample generation (10 jobs, ~2-3 hours, ~$2-3)
poetry run spinlock launch-salad-generation \
    --config configs/distributed/salad_qbm_generation.yaml \
    --verbose

# After completion, download and merge
aws s3 sync s3://spinlock-datasets/qbm_1m_wide/ datasets/parts/ \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com

poetry run spinlock merge-datasets \
    --input "datasets/parts/qbm_1m_wide_part*.h5" \
    --output datasets/qbm_1m_wide.h5 \
    --validate \
    --verbose
```

---

## Alternative: Manage Salad Jobs Manually

If you need to clean up stuck containers:

```bash
cd /home/daniel/projects/spinlock
export $(grep -v '^#' .env | xargs)

# List running container groups
poetry run python3 << 'EOF'
from salad_cloud_sdk import SaladCloudSdk
import os

sdk = SaladCloudSdk(api_key=os.environ.get('SALAD_API_KEY'))

# Try to list container groups (need correct org/project)
try:
    groups = sdk.container_groups.list_container_groups(
        organization_name="YOUR-ORG",
        project_name="YOUR-PROJECT"
    )
    print(f"Container groups: {groups}")
except Exception as e:
    print(f"Error: {e}")
EOF

# Delete a stuck container group
# sdk.container_groups.delete_container_group(
#     organization_name="YOUR-ORG",
#     project_name="YOUR-PROJECT",
#     container_group_name="spinlock-XXXXXXXX-job-0"
# )
```

---

## Key Files Modified

### New Files Created (11)
```
src/spinlock/distributed/salad/base_launcher.py
src/spinlock/distributed/salad/training_launcher.py
src/spinlock/distributed/salad/generation_launcher.py
src/spinlock/cli/launch_salad_generation.py
src/spinlock/cli/merge_datasets.py
configs/experiments/qbm_1m_wide.yaml
configs/distributed/salad_qbm_generation.yaml
configs/distributed/salad_qbm_generation_test.yaml
scripts/test_sobol_offset.py
SALAD_DEPLOYMENT_GUIDE.md
IMPLEMENTATION_SUMMARY.md
```

### Files Modified (8)
```
src/spinlock/sampling/sobol.py               # Sobol offset parameter
src/spinlock/config/schema.py                # SobolConfig.offset field
src/spinlock/cli/generate.py                 # --sobol-offset argument
src/spinlock/dataset/pipeline.py             # Pass offset to sampler
src/spinlock/distributed/salad/launcher.py   # Re-exports for compatibility
src/spinlock/cli/__init__.py                 # Register new commands
docker/training/Dockerfile.salad             # AWS CLI installation
docker/training/entrypoint-salad.sh          # Generation job support + S3 upload
```

---

## Environment Variables Required

In `/home/daniel/projects/spinlock/.env`:
```bash
SALAD_API_KEY=REDACTED_SALAD_API_KEY
DOCKER_USERNAME=psilogon
DOCKER_PASSWORD=REDACTED_DOCKER_PAT
MINIO_ACCESS_KEY=<your-minio-key>
MINIO_SECRET_KEY=<your-minio-secret>
```

MinIO endpoint: https://calculator-cargo-intend-mixer.trycloudflare.com

---

## Expected Results

### Test Deployment (100 samples)
- **Runtime**: 2-3 minutes
- **Cost**: ~$0.01
- **Output**: `s3://spinlock-datasets/qbm_test/qbm_test_part00.h5`
- **Size**: ~150MB compressed

### Full Deployment (1M samples)
- **Runtime**: 2-3 hours (10 parallel jobs)
- **Cost**: ~$2-3
- **Output**: 10 files → `datasets/qbm_1m_wide.h5` (~150GB)
- **Diversity improvement**: 17.1% → 45-60% unique token patterns
- **Zero-variance features**: 78 → <20

---

## Troubleshooting

### If containers fail to start
```bash
# Check Docker Hub for latest image
docker pull psilogon/spinlock:latest

# Verify image has latest code (should see generation_launcher.py)
docker run --rm psilogon/spinlock:latest ls -la src/spinlock/distributed/salad/
```

### If S3 upload fails
```bash
# Test MinIO connectivity
aws s3 ls s3://spinlock-datasets/ \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com

# Create bucket if it doesn't exist
aws s3 mb s3://spinlock-datasets \
    --endpoint-url https://calculator-cargo-intend-mixer.trycloudflare.com
```

### If merge fails with duplicates
This indicates Sobol offsets weren't applied correctly. Check:
- Each job should have unique offset: 0, 100000, 200000, ...
- Verify in Salad dashboard logs: `--sobol-offset` argument is present

---

## Next Steps After Deployment Succeeds

1. **Train VQTokenizer** on 1M diverse dataset:
   ```bash
   poetry run spinlock train-vq-tokenizer \
       --config configs/qbm/vqvae_diverse_v2.yaml \
       --dataset datasets/qbm_1m_wide.h5 \
       --output checkpoints/qbm/vqvae_1m_wide/
   ```

2. **Measure diversity improvement**
   - Expect: 45-60% unique token patterns (vs 17.1% baseline)
   - Zero-variance features: <20 (vs 78 baseline)

3. **Train D3PM** on improved tokens

4. **Bootstrap LLM alignment** with diverse token space

---

## Contact Point

When resuming, you need ONLY ONE piece of information:
- **Correct organization name** from Salad dashboard

Everything else is ready to go! 🚀
