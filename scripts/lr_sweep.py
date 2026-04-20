"""
Learning rate sweep over a base transformer config.

Generates one config per LR, runs them sequentially, then prints a leaderboard.
Derived configs inherit all settings from the base — only lr and experiment_name change.
Early stopping is enabled by default (patience=2) so longer epochs don't waste time.

Usage:
    # Dry run — just generate configs, don't train
    uv run python scripts/lr_sweep.py configs/exp006_xlmr_base.yaml --dry-run

    # Full sweep with default LRs [5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
    TOKENIZERS_PARALLELISM=false uv run python scripts/lr_sweep.py configs/exp006_xlmr_base.yaml

    # Custom LR list
    TOKENIZERS_PARALLELISM=false uv run python scripts/lr_sweep.py configs/exp006_xlmr_base.yaml \\
        --lrs 5e-6 1e-5 3e-5

    # Run in background, follow log
    nohup bash -c 'TOKENIZERS_PARALLELISM=false uv run python scripts/lr_sweep.py configs/exp006_xlmr_base.yaml' \\
        > lr_sweep.log 2>&1 &
    tail -f lr_sweep.log
"""
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Default LR candidates centred around the standard 2e-5
DEFAULT_LRS = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5]

# Applied to all generated sweep configs (safe with early stopping)
SWEEP_EPOCHS = 10
SWEEP_PATIENCE = 2


def _lr_tag(lr: float) -> str:
    """Format lr as a short string safe for filenames, e.g. 2e-5 → lr2e-5."""
    s = f"{lr:.0e}".replace("+", "").replace("0", "", 1) if lr >= 1e-4 else f"{lr:.0e}"
    return f"lr{s}"


def generate_configs(base_config_path: str, lrs: list[float]) -> list[tuple[str, str]]:
    """
    For each lr, write a derived config to configs/ and return (config_path, exp_name) pairs.
    Skips configs that already exist (allows resuming a partial sweep).
    """
    with open(base_config_path) as f:
        base = yaml.safe_load(f)

    base_exp = base["experiment_name"]
    generated = []

    for lr in lrs:
        tag     = _lr_tag(lr)
        exp_name = f"{base_exp}_{tag}"
        out_path = Path("configs") / f"{exp_name}.yaml"

        if out_path.exists():
            print(f"  [skip]  {out_path} already exists — will re-use")
        else:
            cfg = copy.deepcopy(base)
            cfg["experiment_name"] = exp_name
            cfg["notes"] = (
                f"LR sweep on {base_exp}: lr={lr:.1e}  "
                f"(epochs={SWEEP_EPOCHS}, early_stopping_patience={SWEEP_PATIENCE})"
            )
            cfg["model"]["params"]["learning_rate"] = float(lr)
            cfg["model"]["params"]["epochs"] = SWEEP_EPOCHS
            cfg["model"]["params"]["early_stopping_patience"] = SWEEP_PATIENCE
            if "tags" in cfg.get("wandb", {}):
                cfg["wandb"]["tags"] = list(cfg["wandb"]["tags"]) + ["lr-sweep", tag]

            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"  [gen]   {out_path}  (lr={lr:.1e})")

        generated.append((str(out_path), exp_name))

    return generated


def run_sweep(configs: list[tuple[str, str]], uv: str) -> list[str]:
    failed = []
    for cfg_path, exp_name in configs:
        print(f"\n{'='*60}")
        print(f"  START  {exp_name}  (lr={_lr_from_config(cfg_path):.1e})")
        print(f"{'='*60}")
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        result = subprocess.run(
            [uv, "run", "python", "run.py", cfg_path, "--train"],
            env=env,
        )
        if result.returncode != 0:
            print(f"  FAILED  {exp_name}  (exit {result.returncode})")
            failed.append(cfg_path)
        else:
            print(f"  DONE    {exp_name}")
    return failed


def _lr_from_config(path: str) -> float:
    with open(path) as f:
        return yaml.safe_load(f)["model"]["params"]["learning_rate"]


def print_leaderboard(uv: str):
    print(f"\n{'='*60}\n  LEADERBOARD\n{'='*60}")
    subprocess.run([uv, "run", "python", "run.py", "--compare"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_config", help="Path to the base experiment config to sweep over")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS,
                        metavar="LR", help="Learning rates to sweep (default: 5e-6 1e-5 2e-5 3e-5 5e-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs only, do not train")
    parser.add_argument("--uv", default=os.path.expanduser("~/.local/bin/uv"),
                        help="Path to uv binary")
    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        sys.exit(f"Error: config not found: {args.base_config}")

    uv = args.uv
    if not os.path.exists(uv):
        uv = "uv"  # fall back to PATH

    print(f"Base config : {args.base_config}")
    print(f"LR values   : {[f'{lr:.1e}' for lr in args.lrs]}")
    print(f"Epochs      : {SWEEP_EPOCHS} (early stopping patience={SWEEP_PATIENCE})")
    print(f"Dry run     : {args.dry_run}")
    print()

    configs = generate_configs(args.base_config, args.lrs)

    if args.dry_run:
        print("\nDry run complete — configs written, training skipped.")
        print("Run without --dry-run to train.")
        return

    failed = run_sweep(configs, uv)
    print_leaderboard(uv)

    print()
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All sweep experiments completed successfully.")


if __name__ == "__main__":
    main()
