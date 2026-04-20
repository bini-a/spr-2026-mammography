"""
Post-hoc OOF ensemble: average OOF probabilities from multiple experiments,
tune thresholds, and write ensemble artifacts.

Usage (CLI):
    uv run python -m src.ensemble exp004_bertimbau exp010_bertimbau_es exp012_bertimbau_large_es \
        --out exp014_ensemble_v1

Or via run.py (once wired up):
    uv run python run.py configs/exp014_ensemble_v1.yaml --ensemble
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, print_metrics, save_metrics, append_to_results_log
from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def _load_oof(exp_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_probs [N,7], labels [N]) from an experiment directory."""
    path = os.path.join("experiments", exp_dir, "oof_preds.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    df = pd.read_csv(path)
    prob_cols = [f"p{c}" for c in CLASSES]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns in {path}: {missing}")
    probs  = df[prob_cols].values.astype(float)
    labels = df["target"].values.astype(int)
    return probs, labels


def run_ensemble(
    exp_names: list[str],
    out_name: str,
    weights: list[float] | None = None,
    seed: int = 42,
    threshold_n_iter: int = 2000,
    notes: str = "",
) -> str:
    """
    Average OOF probs from exp_names (optionally weighted), tune thresholds,
    write artifacts to experiments/<out_name>/.  Returns the output directory.
    """
    if weights is None:
        weights = [1.0] * len(exp_names)
    if len(weights) != len(exp_names):
        raise ValueError("len(weights) must equal len(exp_names)")

    # Normalise weights
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    print(f"\n{'='*55}")
    print(f"Ensemble   : {out_name}")
    print(f"Components : {exp_names}")
    print(f"Weights    : {w.round(4).tolist()}")
    print(f"{'='*55}")

    all_probs  = []
    labels_ref = None
    for exp, wi in zip(exp_names, w):
        probs, labels = _load_oof(exp)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(
                f"Label mismatch between experiments — are they from the same folds/seed?"
            )
        all_probs.append(probs * wi)
        print(f"  loaded {exp}  ({len(probs):,} rows)")

    avg_probs = np.sum(all_probs, axis=0)  # weighted average (weights already normalised)

    # Raw OOF metrics
    oof_preds   = avg_probs.argmax(axis=1)
    oof_metrics = compute_metrics(labels_ref, oof_preds)
    print_metrics(oof_metrics, title=f"Ensemble OOF (raw) — {out_name}")

    # Threshold tuning
    print("\n--- Threshold Tuning ---")
    offsets = tune_thresholds(avg_probs, labels_ref, n_iter=threshold_n_iter, seed=seed)
    tuned_preds   = (avg_probs + offsets).argmax(axis=1)
    tuned_metrics = compute_metrics(labels_ref, tuned_preds)
    print_metrics(tuned_metrics, title=f"Ensemble OOF (tuned) — {out_name}")

    # Write artifacts
    out_dir = os.path.join("experiments", out_name)
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({"target": labels_ref, "oof_pred": tuned_preds})
    for i, cls in enumerate(CLASSES):
        oof_df[f"p{cls}"] = avg_probs[:, i]
    oof_df.to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)

    save_metrics(oof_metrics, out_dir)
    save_metrics(tuned_metrics, os.path.join(out_dir, "metrics_tuned.json"))
    np.save(os.path.join(out_dir, "thresholds.npy"), offsets)

    # Save ensemble config
    ensemble_cfg = {
        "experiment_name": out_name,
        "type": "ensemble",
        "components": exp_names,
        "weights": w.tolist(),
        "seed": seed,
        "threshold_n_iter": threshold_n_iter,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "ensemble_config.json"), "w") as f:
        json.dump(ensemble_cfg, f, indent=2)

    # Append to results leaderboard using a minimal config-like dict
    fake_config = {
        "model": {"type": "ensemble"},
        "data": {"n_folds": 5},
        "seed": seed,
        "notes": notes or f"Ensemble of: {', '.join(exp_names)}",
    }
    append_to_results_log(out_name, tuned_metrics, fake_config,
                          duration=0, notes=fake_config["notes"])

    print(f"\nOutputs    : experiments/{out_name}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="OOF ensemble from multiple experiments")
    parser.add_argument("experiments", nargs="+", metavar="EXP_NAME",
                        help="Experiment names to ensemble (directories under experiments/)")
    parser.add_argument("--out", required=True, metavar="OUT_NAME",
                        help="Output experiment name (e.g. exp014_ensemble_v1)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-experiment weights (default: uniform)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=2000,
                        help="Threshold tuning iterations (default: 2000)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run_ensemble(
        exp_names=args.experiments,
        out_name=args.out,
        weights=args.weights,
        seed=args.seed,
        threshold_n_iter=args.n_iter,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
