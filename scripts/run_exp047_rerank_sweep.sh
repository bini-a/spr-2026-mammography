#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_EXP="${1:-exp043_5seed_svc_w050}"
SPEC023="${2:-exp047_specialist_023_synth}"
SPEC456="${3:-exp040_specialist_456_synth}"
OUT_PREFIX="${4:-exp048_rerank_023synth}"

A023_VALUES=(0.2 0.3 0.4 0.5)
A456_VALUES=(0.0 0.1 0.2)

echo "=== Rerank sweep ==="
echo "base    : $BASE_EXP"
echo "spec023 : $SPEC023"
echo "spec456 : $SPEC456"
echo "prefix  : $OUT_PREFIX"
echo ""

for a023 in "${A023_VALUES[@]}"; do
  for a456 in "${A456_VALUES[@]}"; do
    out_name="${OUT_PREFIX}_a023_${a023}_a456_${a456}"
    echo "[run] $out_name"
    ./.venv/bin/python -m src.rerank \
      --base "$BASE_EXP" \
      --spec023 "$SPEC023" \
      --spec456 "$SPEC456" \
      --out "$out_name" \
      --alpha023 "$a023" \
      --alpha456 "$a456" \
      --n-iter 2000
  done
done

echo ""
echo "Sweep complete. Check the top of experiments/results.csv for the winners."
