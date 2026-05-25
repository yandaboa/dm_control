#!/usr/bin/env bash
# Create the standalone SMERL conda env (Python 3.11) with CUDA-enabled torch
# matching the host driver (12.6 -> cu128 wheels).
#
# Usage:  bash src/smerl/setup_env.sh
set -euo pipefail

ENV_NAME="${1:-SMERL}"

conda create -n "$ENV_NAME" python=3.11 -y

# Torch first, from the cu128 index, pinned to the same versions as `lti`.
conda run -n "$ENV_NAME" pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0

# Everything else from PyPI.
conda run -n "$ENV_NAME" pip install \
    stable-baselines3==2.8.0 \
    gymnasium==1.2.3 \
    numpy \
    tensorboard

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
