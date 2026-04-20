#!/usr/bin/env bash
# Run experiments sequentially, continue on failure, print leaderboard at the end.
#
# Usage:
#   bash scripts/run_sequential.sh
#   nohup bash scripts/run_sequential.sh > run_sequential.log 2>&1 &
#   tail -f run_sequential.log

set +e   # don't abort on failure — we track failures manually

CONFIGS=(
  configs/exp006_xlmr_base.yaml
  configs/exp007_bertimbau_large.yaml
  configs/exp008_mdeberta.yaml
  configs/exp009_xlmr_large.yaml
)

UV="${UV:-$HOME/.local/bin/uv}"
FAILED=()

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "======================================================="
  echo "  START  $(date '+%Y-%m-%d %H:%M:%S')  $cfg"
  echo "======================================================="
  TOKENIZERS_PARALLELISM=false "$UV" run python run.py "$cfg" --train
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED  $cfg  (exit $rc)"
    FAILED+=("$cfg")
  else
    echo "  DONE    $cfg"
  fi
done

echo ""
echo "======================================================="
echo "  LEADERBOARD"
echo "======================================================="
"$UV" run python run.py --compare

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED experiments:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
  exit 1
else
  echo "All experiments completed successfully."
fi
