#!/usr/bin/env bash
# Adaptive experiment batch — run pivotal experiments first, decide on follow-ups based on results.
#
# PHASE 1 (parallel — ~90min):
#   GPU 0: exp012_bertimbau_large_es  — key question: does large beat base with proper training?
#   GPU 1: exp008_mdeberta            — wildcard: mDeBERTa with FP16 fixed
#
# PHASE 2 (sequential — run whichever of these applies after seeing phase 1 results):
#   exp010_bertimbau_es   — if large ≈ base: squeeze base further with 10ep+ES
#   exp011_bertimbau_ls   — if base wins: test LS on base
#   exp013_bertimbau_large_ls — if large wins: test LS on large
#
# Usage:
#   # Phase 1 (parallel):
#   nohup bash scripts/run_next_batch.sh phase1 > run_next_batch.log 2>&1 &
#
#   # Phase 2 (after seeing results — run the relevant ones):
#   nohup bash scripts/run_next_batch.sh phase2_large_wins > run_next_batch.log 2>&1 &
#   nohup bash scripts/run_next_batch.sh phase2_base_wins  > run_next_batch.log 2>&1 &
#   nohup bash scripts/run_next_batch.sh phase2_tie        > run_next_batch.log 2>&1 &
#
#   tail -f run_next_batch.log

set +e

UV="${UV:-$HOME/.local/bin/uv}"
FAILED=()

run_exp() {
  local cfg=$1
  echo ""
  echo "======================================================="
  echo "  START  $(date '+%Y-%m-%d %H:%M:%S')  $cfg"
  echo "======================================================="
  TOKENIZERS_PARALLELISM=false "$UV" run python run.py "$cfg" --train
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED  $cfg  (exit $rc)"
    FAILED+=("$cfg")
  else
    echo "  DONE    $cfg"
  fi
}

phase=$1

case "$phase" in

  phase1)
    echo "=== PHASE 1: parallel — exp012 (GPU 0) + exp008 (GPU 1) ==="
    # Each config already pins its GPU via 'gpu: 0' / 'gpu: 1'.
    # Do NOT set CUDA_VISIBLE_DEVICES — it remaps device indices and would
    # cause 'gpu: 1' to crash (only device 0 visible after remapping).
    (
      TOKENIZERS_PARALLELISM=false "$UV" run python run.py \
        configs/exp012_bertimbau_large_es.yaml --train
      echo "  exp012 done (GPU 0)"
    ) &
    PID_012=$!

    (
      TOKENIZERS_PARALLELISM=false "$UV" run python run.py \
        configs/exp008_mdeberta.yaml --train
      echo "  exp008 done (GPU 1)"
    ) &
    PID_008=$!

    echo "  exp012 PID=$PID_012  (GPU 0)"
    echo "  exp008 PID=$PID_008  (GPU 1)"
    wait $PID_012 || FAILED+=(configs/exp012_bertimbau_large_es.yaml)
    wait $PID_008 || FAILED+=(configs/exp008_mdeberta.yaml)
    ;;

  phase2_large_wins)
    echo "=== PHASE 2: large won — testing LS on large ==="
    run_exp configs/exp013_bertimbau_large_ls.yaml
    ;;

  phase2_base_wins)
    echo "=== PHASE 2: base won — testing LS on base ==="
    run_exp configs/exp011_bertimbau_ls.yaml
    ;;

  phase2_tie)
    echo "=== PHASE 2: tie — squeeze base with 10ep+ES, then test LS on base ==="
    run_exp configs/exp010_bertimbau_es.yaml
    run_exp configs/exp011_bertimbau_ls.yaml
    ;;

  *)
    echo "Usage: $0 {phase1|phase2_large_wins|phase2_base_wins|phase2_tie}"
    exit 1
    ;;
esac

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
