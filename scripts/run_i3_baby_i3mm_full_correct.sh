#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
CONTRASTIVE_BATCH_SIZE="${CONTRASTIVE_BATCH_SIZE:-1024}"
CONTRASTIVE_SCOPE="${CONTRASTIVE_SCOPE:-full}"
CONTRASTIVE_IMPL="${CONTRASTIVE_IMPL:-memory}"
DIFF_CKPT_DIR="./checkpoint/i3_baby_i3mm_mr0p3"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${DIFF_CKPT_DIR}"

if [[ ! -f "${DIFF_CKPT_DIR}/best_model.pth" ]]; then
  echo "$(date '+%F %T') pretraining i3_baby diffusion checkpoint into ${DIFF_CKPT_DIR}"
  "${PYTHON_BIN}" -u main_pretrain.py \
    --dataset i3_baby \
    --missing_mode i3_mm \
    --MR 0.3 \
    --save_dir "${DIFF_CKPT_DIR}" \
    --epoch "${PRETRAIN_EPOCHS}" \
    --batch_size "${PRETRAIN_BATCH_SIZE}"
else
  echo "$(date '+%F %T') found existing diffusion checkpoint ${DIFF_CKPT_DIR}/best_model.pth"
fi

echo "$(date '+%F %T') starting i3_baby MoDiCF training"
exec "${PYTHON_BIN}" -u main.py \
  --dataset i3_baby \
  --load_dir "${DIFF_CKPT_DIR}/" \
  --missing_mode i3_mm \
  --MR 0.3 \
  --epoch "${TRAIN_EPOCHS}" \
  --early_stop "${EARLY_STOP}" \
  --contrastive_batch_size "${CONTRASTIVE_BATCH_SIZE}" \
  --contrastive_scope "${CONTRASTIVE_SCOPE}" \
  --contrastive_impl "${CONTRASTIVE_IMPL}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --verbose 1
