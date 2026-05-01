"""
Metrics, reporting, and the master results log.
Always report macro F1 + per-class breakdown — never accuracy alone.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = [0, 1, 2, 3, 4, 5, 6]
RESULTS_LOG = "experiments/results.csv"

# Columns shown in --compare (focused on rare/hard classes)
_COMPARE_COLS = [
    "experiment", "macro_f1",
    "f1_c0", "f1_c1", "f1_c3", "f1_c4", "f1_c5", "f1_c6",
    "duration_s", "timestamp", "notes",
]


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    labels = CLASSES if labels is None else list(labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm.tolist(), "labels": labels}


def print_metrics(metrics: dict, title: str = "", labels=None) -> None:
    labels = metrics.get("labels", CLASSES) if labels is None else list(labels)
    if title:
        print(f"\n{'='*55}\n{title}\n{'='*55}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}\n")
    print(f"{'Class':<8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>6}")
    print("-" * 38)
    for cls in labels:
        r = metrics["report"].get(str(cls), {})
        f1  = r.get("f1-score",  0.0)
        pre = r.get("precision", 0.0)
        rec = r.get("recall",    0.0)
        n   = int(r.get("support", 0))
        print(f"{cls:<8} {f1:>6.3f} {pre:>6.3f} {rec:>6.3f} {n:>6}")
    print()


def save_metrics(metrics: dict, out_dir: str) -> None:
    if out_dir.endswith(".json"):
        path = out_dir
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def append_to_results_log(
    exp_name: str,
    metrics: dict,
    config: dict,
    timestamp: str = "",
    duration: float = 0.0,
    notes: str = "",
) -> None:
    """Upsert one row into experiments/results.csv."""
    row = {
        "experiment":  exp_name,
        "macro_f1":    round(metrics["macro_f1"], 4),
        "model_type":  config.get("model", {}).get("type", ""),
        "n_folds":     config.get("data", {}).get("n_folds", ""),
        "seed":        config.get("seed", ""),
        "timestamp":   timestamp,
        "duration_s":  round(duration, 1),
        "notes":       notes or config.get("notes", ""),
    }
    for cls in CLASSES:
        r = metrics["report"].get(str(cls), {})
        row[f"f1_c{cls}"] = round(r.get("f1-score", 0.0), 4)

    df_row = pd.DataFrame([row])
    if os.path.exists(RESULTS_LOG):
        existing = pd.read_csv(RESULTS_LOG, keep_default_na=False)
        existing = existing[existing["experiment"] != exp_name]
        df_out = pd.concat([existing, df_row], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
        df_out = df_row

    df_out = df_out.sort_values("macro_f1", ascending=False)
    df_out.to_csv(RESULTS_LOG, index=False)
    print(f"Results logged → {RESULTS_LOG}")


def print_comparison(full: bool = False) -> None:
    """Print the leaderboard table. full=True shows all columns."""
    if not os.path.exists(RESULTS_LOG):
        print("No experiments logged yet.")
        return

    df = (pd.read_csv(RESULTS_LOG, keep_default_na=False)
            .sort_values("macro_f1", ascending=False)
            .reset_index(drop=True))
    df.index += 1  # rank starts at 1

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{'='*55}")
    print(f"  Experiment leaderboard  (OOF macro F1, descending)")
    print(f"{'='*55}")

    if full:
        print(df.to_string())
    else:
        cols = [c for c in _COMPARE_COLS if c in df.columns]
        print(df[cols].to_string())

    # Highlight the improvement vs the row below
    if len(df) > 1:
        best = df["macro_f1"].iloc[0]
        second = df["macro_f1"].iloc[1]
        print(f"\n  Best vs 2nd: +{best - second:+.4f}  ({df['experiment'].iloc[0]})")
    print()
