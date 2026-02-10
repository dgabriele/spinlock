#!/bin/bash
# Test Docker image before deploying to Salad
# Usage: ./scripts/test_docker_image.sh [image_name]

set -e

IMAGE=${1:-psilogon/spinlock:latest}

echo "========================================"
echo "Testing Docker Image: $IMAGE"
echo "========================================"
echo

# Test 1: Critical imports
echo "Test 1: Verifying critical imports..."
docker run --rm --entrypoint python "$IMAGE" -c "
from experiments.diffusion.models.discrete_d3pm import DiscreteD3PM
import spinlock
print('✓ All critical imports work')
"

# Test 2: CLI available
echo
echo "Test 2: Verifying CLI is available..."
docker run --rm --entrypoint bash "$IMAGE" -c "
poetry run spinlock --version
"

# Test 3: Check experiments directory exists
echo
echo "Test 3: Verifying experiments directory..."
docker run --rm --entrypoint bash "$IMAGE" -c "
ls -la /app/experiments/ | head -n 5
"

# Test 4: Check configs directory exists
echo
echo "Test 4: Verifying configs directory..."
docker run --rm --entrypoint bash "$IMAGE" -c "
ls -la /app/configs/ | head -n 5
"

# Test 5: Verify entry point script
echo
echo "Test 5: Verifying entrypoint script..."
docker run --rm --entrypoint bash "$IMAGE" -c "
ls -la /entrypoint.sh
cat /entrypoint.sh | head -n 10
"

echo
echo "========================================"
echo "✓ All tests passed!"
echo "Image $IMAGE is ready to deploy"
echo "========================================"
