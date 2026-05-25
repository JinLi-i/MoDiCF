#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -u main.py \
  --dataset i3_baby \
  --load_dir ./checkpoint/i3_baby/ \
  --epoch 200 \
  --early_stop 20 \
  --batch_size 2048 \
  --verbose 1
