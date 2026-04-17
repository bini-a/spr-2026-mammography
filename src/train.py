"""
OOF cross-validation training loop.
Called by run.py — do not invoke directly.
"""
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data import load_train, make_folds
from src.evaluate import (
    append_to_results_log,
    compute_metrics,
    print_metrics,
    save_metrics,
)
from src.features import (
    build_features,
    clean_text,
    save_vectorizers,
)
from src.logging_utils import run_log
from src.models.linear import build_model, save_model
from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _probs_to_matrix(probs, model_classes):
    """Map model.predict_proba output to a (n, 7) matrix aligned to CLASSES."""
    matrix = np.zeros((len(probs), len(CLASSES)))
    for i, cls in enumerate(model_classes):
        matrix[:, CLASSES.index(int(cls))] = probs[:, i]
    return matrix


def _step(pbar, msg):
    pbar.set_postfix_str(msg, refresh=True)


def run_training(config_path: str, notes_override: str = None) -> tuple:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    out_dir  = os.path.join("experiments", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    dest = os.path.join(out_dir, "config.yaml")
    if os.path.abspath(config_path) != os.path.abspath(dest):
        shutil.copy(config_path, dest)

    # Notes: CLI override > config field > empty
    notes = notes_override or config.get("notes", "")

    with run_log(out_dir):
        _run(config, exp_name, out_dir, config_path, notes)

    return out_dir


def _run(config, exp_name, out_dir, config_path, notes):
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    seed      = config.get("seed", 42)
    n_folds   = config["data"]["n_folds"]
    feat_cfg  = config.get("features", {})
    model_cfg = config["model"]

    print(f"\n{'='*55}")
    print(f"Experiment : {exp_name}")
    print(f"Config     : {config_path}")
    print(f"Started    : {timestamp}")
    print(f"Git commit : {_git_hash()}")
    if notes:
        print(f"Notes      : {notes}")
    print(f"{'='*55}")

    # ── Load & prepare data ──────────────────────────────
    print("Loading data...", end=" ", flush=True)
    df = load_train()
    df["text"] = df["report"].apply(clean_text)
    df = make_folds(df, n_folds=n_folds, seed=seed)
    print(f"done  ({len(df):,} rows, {n_folds} folds)")

    oof_probs = np.zeros((len(df), len(CLASSES)))
    fold_f1s  = []

    # ── OOF cross-validation ─────────────────────────────
    fold_bar = tqdm(range(n_folds), desc="CV folds", unit="fold", ncols=70)
    for fold in fold_bar:
        train_mask = df["fold"] != fold
        val_mask   = df["fold"] == fold
        val_idx    = df.index[val_mask].tolist()

        n_tr = train_mask.sum()
        n_val = val_mask.sum()
        fold_bar.set_description(f"Fold {fold+1}/{n_folds}  ({n_tr}tr/{n_val}val)")

        _step(fold_bar, "vectorizing")
        X_train, X_val, vectorizers = build_features(
            df.loc[train_mask, "text"].tolist(),
            df.loc[val_mask,   "text"].tolist(),
            feat_cfg,
        )
        y_train = df.loc[train_mask, "target"].values
        y_val   = df.loc[val_mask,   "target"].values

        _step(fold_bar, "fitting")
        model = build_model(model_cfg)
        model.fit(X_train, y_train)

        _step(fold_bar, "scoring")
        val_probs   = model.predict_proba(X_val)
        prob_matrix = _probs_to_matrix(val_probs, model.classes_)
        for i, pos in enumerate(val_idx):
            oof_probs[pos] = prob_matrix[i]

        fold_preds   = np.argmax(prob_matrix, axis=1)
        fold_metrics = compute_metrics(y_val, fold_preds)
        fold_f1s.append(fold_metrics["macro_f1"])
        fold_bar.set_postfix({"F1": f"{fold_metrics['macro_f1']:.4f}"}, refresh=True)

    fold_bar.close()

    # ── OOF evaluation ───────────────────────────────────
    oof_preds   = np.argmax(oof_probs, axis=1)
    oof_metrics = compute_metrics(df["target"].values, oof_preds)
    print_metrics(oof_metrics, title=f"OOF Results — {exp_name}")
    print(f"Fold F1s : {[round(f, 4) for f in fold_f1s]}")
    print(f"Mean ± σ : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    # Save OOF predictions (used for threshold tuning and error analysis)
    oof_df = df[["ID", "target", "fold"]].copy()
    oof_df["oof_pred"] = oof_preds
    for i, cls in enumerate(CLASSES):
        oof_df[f"p{cls}"] = oof_probs[:, i]
    oof_df.to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)
    save_metrics(oof_metrics, out_dir)

    # ── Optional threshold tuning ────────────────────────
    if config.get("threshold", {}).get("enabled", False):
        print("\n--- Threshold Tuning ---")
        offsets = tune_thresholds(oof_probs, df["target"].values, seed=seed)
        np.save(os.path.join(out_dir, "thresholds.npy"), offsets)

        tuned_preds   = np.argmax(oof_probs + offsets, axis=1)
        tuned_metrics = compute_metrics(df["target"].values, tuned_preds)
        print_metrics(tuned_metrics, title="OOF Results (after threshold tuning)")
        save_metrics(tuned_metrics, os.path.join(out_dir, "metrics_tuned.json"))
        oof_metrics = tuned_metrics

    # ── Retrain on full data ─────────────────────────────
    print("\n--- Retraining on full data ---")
    with tqdm(total=3, desc="Full retrain", unit="step", ncols=70) as pbar:
        pbar.set_postfix_str("vectorizing", refresh=True)
        X_full, vectorizers_full = build_features(df["text"].tolist(), config=feat_cfg)
        pbar.update(1)

        pbar.set_postfix_str("fitting", refresh=True)
        model_full = build_model(model_cfg)
        model_full.fit(X_full, df["target"].values)
        pbar.update(1)

        pbar.set_postfix_str("saving", refresh=True)
        save_model(model_full, out_dir)
        save_vectorizers(vectorizers_full, out_dir)
        pbar.update(1)

    duration = time.time() - t_start
    print(f"\nDuration   : {duration:.0f}s  ({duration/60:.1f}m)")
    append_to_results_log(exp_name, oof_metrics, config,
                          timestamp=timestamp, duration=duration, notes=notes)
    print(f"Outputs    : experiments/{exp_name}/")
