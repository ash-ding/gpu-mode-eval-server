#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_DIR="$PROJECT_ROOT/container"

IMAGE_NAME="${1:-eval-server}"
IMAGE_TAG="${2:-latest}"

# Detect container runtime
if command -v podman &>/dev/null; then
    RUNTIME=podman
elif command -v docker &>/dev/null; then
    RUNTIME=docker
else
    echo "ERROR: No container runtime found. Install podman or docker."
    exit 1
fi

echo "Building $IMAGE_NAME:$IMAGE_TAG with $RUNTIME..."
$RUNTIME build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$CONTAINER_DIR/Dockerfile" "$CONTAINER_DIR"

echo "Verifying image..."
$RUNTIME run --rm "$IMAGE_NAME:$IMAGE_TAG" python3 -c "
import torch
import triton
import numpy
print(f'PyTorch: {torch.__version__}')
print(f'Triton: {triton.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy: {numpy.__version__}')
print('Container image verified.')
"

echo "Build complete: $IMAGE_NAME:$IMAGE_TAG"
