"""
Soft specialist reranking: blend a base 7-class experiment with one or more
specialist experiments on selected confusion groups, then optionally tune
thresholds and write a standard experiment directory.

Usage:
    uv run python -m src.rerank \
        --base exp025_multiseed_ensemble \
        --spec023 exp034a_023_specialist \
        --spec456 exp034b_456_specialist \
        --out exp034_rerank_v1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.evaluate import append_to_results_log, compute_metrics, print_metrics, save_metrics
from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def _load_oof(exp_name: str) -> pd.DataFrame:
    path = os.path.join("experiments", exp_name, "oof_preds.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    df = pd.read_csv(path)
    needed = ["target"] + [f"p{c}" for c in CLASSES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if "ID" not in df.columns:
        df["ID"] = np.arange(len(df))
    return df[["ID", "target"] + [f"p{c}" for c in CLASSES]].copy()


def _load_spec_probs(exp_name: str, n_rows: int) -> np.ndarray:
    """Load specialist OOF prob columns as a (n_rows, 7) array by row position."""
    path = os.path.join("experiments", exp_name, "oof_preds.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    df = pd.read_csv(path, usecols=[f"p{c}" for c in CLASSES])
    if len(df) != n_rows:
        raise ValueError(
            f"Row count mismatch: base has {n_rows} rows but {path} has {len(df)}"
        )
    return df[[f"p{c}" for c in CLASSES]].values.astype(float)


def _renorm_group(probs: np.ndarray, group: list[int]) -> np.ndarray:
    sub = probs[:, group].copy()
    denom = sub.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return sub / denom


def _blend_group(base_probs: np.ndarray, spec_probs: np.ndarray, group: list[int], alpha: float) -> np.ndarray:
    out = base_probs.copy()
    base_mass = base_probs[:, group].sum(axis=1, keepdims=True)
    base_group = _renorm_group(base_probs, group)
    spec_group = _renorm_group(spec_probs, group)
    blended = (1.0 - alpha) * base_group + alpha * spec_group
    out[:, group] = blended * base_mass
    return out


def run_rerank(
    base_exp: str,
    out_name: str,
    spec023_exp: str | None = None,
    spec456_exp: str | None = None,
    alpha023: float = 0.5,
    alpha456: float = 0.5,
    threshold_n_iter: int = 2000,
    seed: int = 42,
    notes: str = "",
) -> str:
    base_df = _load_oof(base_exp)
    work = base_df.copy()
    probs = work[[f"p{c}" for c in CLASSES]].values.astype(float)

    n_rows = len(base_df)

    if spec023_exp:
        spec023_probs = _load_spec_probs(spec023_exp, n_rows)
        top2 = np.argsort(probs, axis=1)[:, -2:]
        gate023 = np.isin(top2, [0, 2, 3]).all(axis=1)
        blended = _blend_group(probs, spec023_probs, [0, 2, 3], alpha023)
        probs[gate023] = blended[gate023]

    if spec456_exp:
        spec456_probs = _load_spec_probs(spec456_exp, n_rows)
        top2 = np.argsort(probs, axis=1)[:, -2:]
        gate446 = np.isin(top2, [4, 5, 6]).all(axis=1)
        blended = _blend_group(probs, spec456_probs, [4, 5, 6], alpha456)
        probs[gate446] = blended[gate446]

    raw_preds = probs.argmax(axis=1)
    raw_metrics = compute_metrics(base_df["target"].values, raw_preds)
    print_metrics(raw_metrics, title=f"Rerank OOF (raw) — {out_name}")

    print("\n--- Threshold Tuning ---")
    offsets = tune_thresholds(probs, base_df["target"].values, n_iter=threshold_n_iter, seed=seed)
    tuned_preds = (probs + offsets).argmax(axis=1)
    tuned_metrics = compute_metrics(base_df["target"].values, tuned_preds)
    print_metrics(tuned_metrics, title=f"Rerank OOF (tuned) — {out_name}")

    out_dir = os.path.join("experiments", out_name)
    os.makedirs(out_dir, exist_ok=True)

    oof_df = base_df[["ID", "target"]].reset_index(drop=True).copy()
    oof_df["oof_pred"] = tuned_preds
    for i, cls in enumerate(CLASSES):
        oof_df[f"p{cls}"] = probs[:, i]
    oof_df.to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)

    save_metrics(raw_metrics, out_dir)
    save_metrics(tuned_metrics, os.path.join(out_dir, "metrics_tuned.json"))
    np.save(os.path.join(out_dir, "thresholds.npy"), offsets)

    rerank_cfg = {
        "experiment_name": out_name,
        "type": "rerank",
        "base": base_exp,
        "spec023": spec023_exp,
        "spec456": spec456_exp,
        "alpha023": alpha023,
        "alpha456": alpha456,
        "threshold_n_iter": threshold_n_iter,
        "seed": seed,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "rerank_config.json"), "w") as f:
        json.dump(rerank_cfg, f, indent=2)

    fake_config = {
        "model": {"type": "ensemble"},
        "data": {"n_folds": 5},
        "seed": seed,
        "notes": notes or f"Rerank of base={base_exp}, spec023={spec023_exp}, spec456={spec456_exp}",
    }
    append_to_results_log(out_name, tuned_metrics, fake_config, duration=0, notes=fake_config["notes"])
    print(f"\nOutputs    : experiments/{out_name}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Soft specialist reranking from OOF probabilities")
    parser.add_argument("--base", required=True, help="Base experiment name")
    parser.add_argument("--spec023", default=None, help="Specialist experiment for classes {0,2,3}")
    parser.add_argument("--spec456", default=None, help="Specialist experiment for classes {4,5,6}")
    parser.add_argument("--out", required=True, help="Output experiment name")
    parser.add_argument("--alpha023", type=float, default=0.5)
    parser.add_argument("--alpha456", type=float, default=0.5)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run_rerank(
        base_exp=args.base,
        out_name=args.out,
        spec023_exp=args.spec023,
        spec456_exp=args.spec456,
        alpha023=args.alpha023,
        alpha456=args.alpha456,
        threshold_n_iter=args.n_iter,
        seed=args.seed,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
