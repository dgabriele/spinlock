#!/bin/bash
set -e

echo "[Salad Container] Starting..."

# Environment variables set by Salad launcher:
# - S3_ENDPOINT_URL: MinIO or S3 endpoint
# - S3_BUCKET: Bucket name
# - S3_OUTPUT_PATH: Full S3 path for generation output (generation only)
# - DATASET_PATH: Dataset S3 key (training only)
# - CHECKPOINT_PATH: Checkpoint S3 key prefix (training only)
# - AWS_ACCESS_KEY_ID: S3 credentials
# - AWS_SECRET_ACCESS_KEY: S3 credentials
# - RANK: Process rank (training only)
# - WORLD_SIZE: Total processes (training only)
# - JOB_INDEX: Job index (generation only)

# Detect job type
if [ -n "$JOB_INDEX" ]; then
    JOB_TYPE="generation"
    echo "[Salad] Job type: Generation (Job $JOB_INDEX)"
else
    JOB_TYPE="training"
    echo "[Salad] Job type: Training (Rank $RANK/$WORLD_SIZE)"

    # Set up distributed environment for training
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=INFO
fi

# Run the spinlock command
echo "[Salad] Running: poetry run spinlock $@"
poetry run spinlock "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[Salad] ✗ Command failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "[Salad] ✓ Command completed successfully"

# For generation jobs: upload result to S3
if [ "$JOB_TYPE" = "generation" ] && [ -n "$S3_OUTPUT_PATH" ]; then
    echo "[Salad] Uploading result to S3..."

    # Extract output file from command args (look for --output)
    OUTPUT_FILE=""
    for i in "$@"; do
        if [ "$PREV_ARG" = "--output" ]; then
            OUTPUT_FILE="$i"
            break
        fi
        PREV_ARG="$i"
    done

    if [ -z "$OUTPUT_FILE" ]; then
        echo "[Salad] ✗ Could not determine output file from command args"
        exit 1
    fi

    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "[Salad] ✗ Output file not found: $OUTPUT_FILE"
        exit 1
    fi

    # Upload using aws CLI (should be installed in Docker image)
    echo "[Salad]   Source: $OUTPUT_FILE"
    echo "[Salad]   Destination: s3://$S3_BUCKET/$S3_OUTPUT_PATH"
    echo "[Salad]   Endpoint: $S3_ENDPOINT_URL"

    aws s3 cp "$OUTPUT_FILE" "s3://$S3_BUCKET/$S3_OUTPUT_PATH" \
        --endpoint-url "$S3_ENDPOINT_URL" \
        --no-verify-ssl

    if [ $? -eq 0 ]; then
        echo "[Salad] ✓ Upload successful"
    else
        echo "[Salad] ✗ Upload failed"
        exit 1
    fi
fi

echo "[Salad] Container exiting"
