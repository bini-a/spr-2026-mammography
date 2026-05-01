#!/usr/bin/env python3
"""
Compact status dashboard for the final experiment push.

Usage:
    ./.venv/bin/python scripts/final_push_status.py
    watch -n 30 ./.venv/bin/python scripts/final_push_status.py
"""
from __future__ import annotations

import csv
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "experiments" / "results.csv"
RUN_DIR = ROOT / "logs" / "final_push"
MANIFEST = RUN_DIR / "launch_manifest.tsv"
DEFAULT_WATCH = [
    "exp044_rerank_a023_0.3_a456_0.1",
    "exp043_5seed_svc_w050",
    "exp035_svc_best",
    "exp045_rdrop",
    "exp046_awp",
    "exp047_specialist_023_synth",
]


def _read_manifest() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not MANIFEST.exists():
        return rows
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows[row["experiment"]] = row
    return rows


def _pid_alive(pid: str | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _age_str(iso_text: str | None) -> str:
    if not iso_text:
        return "-"
    try:
        started = datetime.fromisoformat(iso_text)
    except ValueError:
        return "-"
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - started.astimezone(timezone.utc)
    mins = int(delta.total_seconds() // 60)
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m" if hours else f"{mins}m"


def _tail_lines(path: Path, n: int = 2) -> str:
    if not path.exists():
        return "-"
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "-"
    return " | ".join(lines[-n:])


def _detect_log(exp: str) -> Path | None:
    train_log = ROOT / "experiments" / exp / "train.log"
    if train_log.exists():
        return train_log
    launcher_log = RUN_DIR / f"{exp}.launcher.log"
    if launcher_log.exists():
        return launcher_log
    return None


def _recently_updated(path: Path | None, max_age_minutes: int = 20) -> bool:
    if path is None or not path.exists():
        return False
    age_s = datetime.now().timestamp() - path.stat().st_mtime
    return age_s <= max_age_minutes * 60


def _status_row(exp: str, results: pd.DataFrame, manifest: dict[str, dict[str, str]]) -> dict[str, str]:
    row = manifest.get(exp, {})
    pid = row.get("pid")
    running = _pid_alive(pid)
    log_path = _detect_log(exp)
    has_metrics = exp in results.index

    if running:
        status = "running"
    elif has_metrics:
        status = "done"
    elif _recently_updated(log_path):
        status = "started"
    elif (ROOT / "configs" / f"{exp}.yaml").exists():
        status = "planned"
    else:
        status = "missing"

    metric = f"{results.loc[exp, 'macro_f1']:.4f}" if has_metrics else "-"
    gpu = row.get("gpu", "-")
    age = _age_str(row.get("started_at"))
    log_tail = _tail_lines(log_path) if log_path else "-"

    return {
        "experiment": exp,
        "status": status,
        "gpu": gpu,
        "pid": pid or "-",
        "age": age,
        "macro_f1": metric,
        "log": log_tail,
    }


def main() -> None:
    manifest = _read_manifest()
    watch = list(dict.fromkeys(DEFAULT_WATCH + list(manifest)))

    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV, keep_default_na=False).set_index("experiment")
    else:
        results = pd.DataFrame()

    print("\n=== Final Push Status ===")
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run dir : {RUN_DIR}")

    if not results.empty:
        top = results.reset_index()[["experiment", "macro_f1", "f1_c3", "f1_c4", "f1_c5", "f1_c6"]].head(10)
        print("\nTop leaderboard snapshot:")
        print(top.to_string(index=False))

    rows = [_status_row(exp, results, manifest) for exp in watch]
    print("\nTracked experiments:")
    for row in rows:
        print(
            f"{row['experiment']:<32} "
            f"{row['status']:<8} "
            f"gpu={row['gpu']:<2} "
            f"pid={row['pid']:<7} "
            f"age={row['age']:<6} "
            f"f1={row['macro_f1']:<7} "
            f"{row['log']}"
        )

    print("\nMonitor commands:")
    print("  watch -n 30 ./.venv/bin/python scripts/final_push_status.py")
    print("  tail -f experiments/<exp_name>/train.log")
    print("  tail -f logs/final_push/<exp_name>.launcher.log")


if __name__ == "__main__":
    main()
