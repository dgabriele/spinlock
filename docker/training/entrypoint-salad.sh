#!/bin/bash
set -e

echo "[Salad Container] Starting..."

# Environment variables set by Salad launcher:
# - S3_ENDPOINT_URL: MinIO or S3 endpoint
# - S3_BUCKET: Bucket name
# - DATASET_PATH: Dataset S3 key
# - CHECKPOINT_PATH: Checkpoint S3 key prefix
# - AWS_ACCESS_KEY_ID: S3 credentials
# - AWS_SECRET_ACCESS_KEY: S3 credentials
# - RANK: Process rank
# - WORLD_SIZE: Total processes
# - MASTER_PORT: NCCL port

# Note: Data sync is now handled by Python code in train_meta_operator.py
# via _sync_data_from_cloud() method, so we don't need to download here.

# Set up distributed environment
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# Run training command
echo "[Salad] Starting training (Rank $RANK/$WORLD_SIZE)..."
exec python -m spinlock.cli "$@"
