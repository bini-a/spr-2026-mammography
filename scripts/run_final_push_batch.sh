#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="logs/final_push"
MANIFEST="$RUN_DIR/launch_manifest.tsv"
mkdir -p "$RUN_DIR"

if [ ! -f "$MANIFEST" ]; then
  printf "experiment\tconfig\tgpu\tpid\tstarted_at\n" > "$MANIFEST"
fi

launch_train() {
  local exp_name="$1"
  local config_path="$2"
  local physical_gpu="$3"
  local launcher_log="$RUN_DIR/${exp_name}.launcher.log"
  local pid_file="$RUN_DIR/${exp_name}.pid"

  if [ -f "$pid_file" ]; then
    local existing_pid
    existing_pid="$(cat "$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "[skip] $exp_name already running (pid=$existing_pid)"
      return
    fi
  fi

  echo "[launch] $exp_name on physical GPU $physical_gpu"
  echo "         config: $config_path"
  echo "         logs  : $launcher_log and experiments/$exp_name/train.log"

  nohup env CUDA_VISIBLE_DEVICES="$physical_gpu" ./.venv/bin/python run.py "$config_path" --train \
    > "$launcher_log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$exp_name" "$config_path" "$physical_gpu" "$pid" "$(date -Iseconds)" >> "$MANIFEST"
  echo "[ok] started $exp_name (pid=$pid)"
}

case "${1:-batch1}" in
  batch1)
    launch_train "exp047_specialist_023_synth" "configs/exp047_specialist_023_synth.yaml" "0"
    launch_train "exp045_rdrop" "configs/exp045_rdrop.yaml" "1"
    ;;
  batch2)
    launch_train "exp046_awp" "configs/exp046_awp.yaml" "1"
    ;;
  status)
    ./.venv/bin/python scripts/final_push_status.py
    ;;
  *)
    echo "Usage: bash scripts/run_final_push_batch.sh [batch1|batch2|status]"
    exit 1
    ;;
esac
