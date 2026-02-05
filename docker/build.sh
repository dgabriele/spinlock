#!/bin/bash
# Docker build helper for Spinlock
set -e

REGISTRY="${DOCKER_REGISTRY:-psilogon}"
VERSION="${VERSION:-latest}"

usage() {
    echo "Usage: $0 [base|salad|all]"
    echo ""
    echo "Examples:"
    echo "  $0 base              # Build base image only"
    echo "  $0 salad             # Build Salad training image"
    echo "  $0 all               # Build all images"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_REGISTRY      Docker registry (default: psilogon)"
    echo "  VERSION              Image version tag (default: latest)"
    exit 1
}

build_base() {
    echo "=== Building base image ==="
    docker build \
        -f docker/base/Dockerfile \
        -t spinlock:base \
        -t "$REGISTRY/spinlock:base-$VERSION" \
        .
    echo "✓ Base image built"
}

build_salad() {
    echo "=== Building Salad training image ==="

    # First ensure base image exists
    if ! docker images spinlock:base | grep -q spinlock; then
        echo "Base image not found, building it first..."
        build_base
    fi

    docker build \
        -f docker/training/Dockerfile.salad \
        -t spinlock:salad \
        -t "$REGISTRY/spinlock:salad-$VERSION" \
        -t "$REGISTRY/spinlock:latest" \
        .
    echo "✓ Salad training image built"
}

case "${1:-help}" in
    base)
        build_base
        ;;
    salad)
        build_salad
        ;;
    all)
        build_base
        build_salad
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "=== Images built ==="
docker images | grep spinlock

echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push $REGISTRY/spinlock:base-$VERSION"
echo "  docker push $REGISTRY/spinlock:salad-$VERSION"
echo "  docker push $REGISTRY/spinlock:latest"
