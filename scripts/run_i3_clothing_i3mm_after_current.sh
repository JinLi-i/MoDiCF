#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WAIT_PID="${WAIT_PID-579651}"
GPU_ID="${GPU_ID:-0,1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
SEED="${SEED:-1}"
MISSING_SEED="${MISSING_SEED:-0}"
EARLY_STOP="${EARLY_STOP:-20}"
EARLY_STOP_SPLIT="${EARLY_STOP_SPLIT:-val}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
CONTRASTIVE_BATCH_SIZE="${CONTRASTIVE_BATCH_SIZE:-1024}"
CONTRASTIVE_SCOPE="${CONTRASTIVE_SCOPE:-full}"
CONTRASTIVE_IMPL="${CONTRASTIVE_IMPL:-exact}"
CONTRASTIVE_DEVICES="${CONTRASTIVE_DEVICES:-0,1}"
DIFF_CKPT_DIR="${DIFF_CKPT_DIR:-./checkpoint/i3_clothing_i3mm_mr0p3_missingseed${MISSING_SEED}_seed${SEED}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for PID ${WAIT_PID} before starting i3_clothing on GPU ${GPU_ID}"
    sleep 60
  done
fi

mkdir -p "${DIFF_CKPT_DIR}"

if [[ ! -f "${DIFF_CKPT_DIR}/best_model.pth" ]]; then
  echo "$(date '+%F %T') pretraining i3_clothing diffusion checkpoint into ${DIFF_CKPT_DIR}"
  "${PYTHON_BIN}" -u main_pretrain.py \
    --dataset i3_clothing \
    --missing_mode i3_mm \
    --MR 0.3 \
    --seed "${SEED}" \
    --missing_seed "${MISSING_SEED}" \
    --save_dir "${DIFF_CKPT_DIR}" \
    --epoch "${PRETRAIN_EPOCHS}" \
    --batch_size "${PRETRAIN_BATCH_SIZE}"
else
  echo "$(date '+%F %T') found existing diffusion checkpoint ${DIFF_CKPT_DIR}/best_model.pth"
fi

echo "$(date '+%F %T') starting i3_clothing MoDiCF training"
exec "${PYTHON_BIN}" -u main.py \
  --dataset i3_clothing \
  --load_dir "${DIFF_CKPT_DIR}/" \
  --missing_mode i3_mm \
  --MR 0.3 \
  --seed "${SEED}" \
  --missing_seed "${MISSING_SEED}" \
  --epoch "${TRAIN_EPOCHS}" \
  --early_stop "${EARLY_STOP}" \
  --early_stop_split "${EARLY_STOP_SPLIT}" \
  --contrastive_batch_size "${CONTRASTIVE_BATCH_SIZE}" \
  --contrastive_scope "${CONTRASTIVE_SCOPE}" \
  --contrastive_impl "${CONTRASTIVE_IMPL}" \
  --contrastive_devices "${CONTRASTIVE_DEVICES}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --verbose 1
