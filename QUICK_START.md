# Salad.com Training - Quick Start Guide

## ⚡ Quick Reference

### Before First Use

```bash
# 1. Update configs with your Salad organization name
vim configs/50k_baseline/mno/salad_4x4070.yaml
vim configs/50k_baseline/mno/salad_test.yaml
# Change: organization: "your-org-name"

# 2. Add Docker Hub password to .env
vim .env
# Update: DOCKER_PASSWORD=<your_password>

# 3. Build and push Docker images
./docker/build.sh all
docker login
docker push psilogon/spinlock:latest

# 4. (Optional) Make MinIO persistent
sudo bash /tmp/install-minio-services.sh
```

### Test Run ($0.06 for 1 hour)

```bash
poetry run spinlock train-meta-operator \
  --config configs/50k_baseline/mno/salad_test.yaml
```

### Production Run ($4.48 for 16 hours)

```bash
poetry run spinlock train-meta-operator \
  --config configs/50k_baseline/mno/salad_4x4070.yaml
```

## 📊 System Status

### MinIO
- **Status**: ✅ Running (PID: 495802)
- **Dataset**: ✅ Uploaded (21.47 GB)
- **Endpoint**: `http://67.87.172.104:9000`
- **Bucket**: `spinlock-training`

### Salad Integration
- **Status**: ✅ Installed and verified
- **Configs**:
  - Test: `configs/50k_baseline/mno/salad_test.yaml`
  - Production: `configs/50k_baseline/mno/salad_4x4070.yaml`

### Docker
- **Base Image**: ✅ Ready to build
- **Salad Image**: ✅ Ready to build
- **Build Script**: `./docker/build.sh`

## 🔑 Credentials

All stored in `.env`:
- `SALAD_API_KEY` - Already set
- `DOCKER_USERNAME` - Already set (psilogon)
- `DOCKER_PASSWORD` - **Need to set**
- `MINIO_ENDPOINT` - Already set
- `MINIO_ACCESS_KEY` - Already set
- `MINIO_SECRET_KEY` - Already set

## 💰 Cost Breakdown

| Configuration | GPUs | Cost/hr | Duration | Total |
|--------------|------|---------|----------|-------|
| Test         | 1× RTX 3060 | $0.06 | ~1 hr | ~$0.06 |
| Production   | 4× RTX 4070 | $0.28 | ~16 hr | ~$4.48 |

**Storage**: $0 (local MinIO)

## 📚 Full Documentation

See `SALAD_INTEGRATION_SUMMARY.md` for complete implementation details.
