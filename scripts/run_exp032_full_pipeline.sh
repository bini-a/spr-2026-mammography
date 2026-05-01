#!/usr/bin/env bash
#
# Full sequential launcher for:
#   1. exp032 supervised synthetic pretrain (overwrites models/bertimbau_synth_cls)
#   2. exp032 real-data fine-tune
#   3. exp034a specialist on {0,2,3}
#   4. exp034b specialist on {4,5,6}
#   5. exp034 soft specialist rerank
#
# Usage:
#   nohup bash scripts/run_exp032_full_pipeline.sh > run_exp032_full_pipeline.log 2>&1 &
#   BASE_EXP=exp025_multiseed_ensemble nohup bash scripts/run_exp032_full_pipeline.sh > run_exp032_full_pipeline.log 2>&1 &
#
# Optional environment variables:
#   BASE_EXP       Base 7-class experiment used by rerank
#   RERANK_OUT     Output experiment name for rerank stage
#   CUDA_DEVICES   GPU list for training stages (default: 0,1)
#   UV_CACHE_DIR   UV cache dir (default: /tmp/uv-cache)

set -euo pipefail

BASE_EXP="${BASE_EXP:-exp025_multiseed_ensemble}"
RERANK_OUT="${RERANK_OUT:-exp034_rerank_v1}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

PRETRAIN_DIR="models/bertimbau_synth_cls"
LOG_DIR="logs"

mkdir -p "${LOG_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo ""
  echo "[$(timestamp)] === ${name} ==="
  echo "[$(timestamp)] Log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

echo "[$(timestamp)] Starting full exp032 -> exp034 pipeline"
echo "[$(timestamp)] BASE_EXP=${BASE_EXP}"
echo "[$(timestamp)] RERANK_OUT=${RERANK_OUT}"
echo "[$(timestamp)] CUDA_DEVICES=${CUDA_DEVICES}"
echo "[$(timestamp)] UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "[$(timestamp)] Pretrain checkpoint will be refreshed at ${PRETRAIN_DIR}"

if [ ! -f "experiments/${BASE_EXP}/oof_preds.csv" ]; then
  echo "[$(timestamp)] ERROR: Missing base OOF predictions at experiments/${BASE_EXP}/oof_preds.csv"
  exit 1
fi

run_step "exp032_pretrain" env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  TOKENIZERS_PARALLELISM=false \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv run python scripts/synthetic_cls_pretrain.py \
    --out "${PRETRAIN_DIR}"

run_step "exp032_ft" env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  TOKENIZERS_PARALLELISM=false \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv run python run.py configs/exp032_bertimbau_synthcls_ft.yaml --train

run_step "exp034a_023_specialist" env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  TOKENIZERS_PARALLELISM=false \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv run python run.py configs/exp034a_023_specialist.yaml --train

run_step "exp034b_456_specialist" env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  TOKENIZERS_PARALLELISM=false \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv run python run.py configs/exp034b_456_specialist.yaml --train

run_step "exp034_rerank" env \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  uv run python -m src.rerank \
    --base "${BASE_EXP}" \
    --spec023 exp034a_023_specialist \
    --spec456 exp034b_456_specialist \
    --out "${RERANK_OUT}"

echo ""
echo "[$(timestamp)] Full pipeline completed successfully"
echo "[$(timestamp)] Outputs:"
echo "  ${PRETRAIN_DIR}/"
echo "  experiments/exp032_bertimbau_synthcls_ft/"
echo "  experiments/exp034a_023_specialist/"
echo "  experiments/exp034b_456_specialist/"
echo "  experiments/${RERANK_OUT}/"
