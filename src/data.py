"""
Data loading and fold creation.
Handles both local and Kaggle runtime paths automatically.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Searched in order - first match wins
_DATA_DIRS = [
    "/kaggle/input/competitions/spr-2026-mammography-report-classification",
    "/kaggle/input/spr-2026-mammography-report-classification",
    "./spr-2026-mammography-report-classification",
    "../spr-2026-mammography-report-classification",
    "../../spr-2026-mammography-report-classification",
]


def find_data_dir() -> str:
    env_data_dir = os.getenv("SPR2026_DATA_DIR")
    if env_data_dir:
        env_path = Path(env_data_dir)
        if (env_path / "train.csv").exists():
            return str(env_path)

    tried = []
    for data_dir in _DATA_DIRS:
        train_path = Path(data_dir) / "train.csv"
        tried.append(str(train_path))
        if train_path.exists():
            return str(Path(data_dir))

    raise FileNotFoundError(
        "Could not find data directory. Checked these train.csv paths:\n"
        + "\n".join(f"- {path}" for path in tried)
        + "\nSet SPR2026_DATA_DIR or place the competition data in "
        "./spr-2026-mammography-report-classification/."
    )


def load_train(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    df = pd.read_csv(Path(data_dir) / "train.csv")
    df.columns = df.columns.str.strip()
    df["target"] = df["target"].astype(int)
    return df


def load_test(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    path = Path(data_dir) / "test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"test.csv not found at {path}.\n"
            "This is expected locally - it only exists in the Kaggle evaluation runtime."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
               group_aware: bool = True) -> pd.DataFrame:
    """Add a 'fold' column (0..n_folds-1).

    group_aware=True (default): identical report texts are guaranteed to land in
    the same fold, eliminating leakage from the ~54% duplicate rows.
    Stratification is on the majority label per text group.
    """
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1

    if group_aware:
        unique_texts = df["report"].unique()
        text_to_gid  = {t: i for i, t in enumerate(unique_texts)}
        groups       = df["report"].map(text_to_gid).values

        # Stratify on majority label per group (handles 11 conflicting-label groups)
        majority_label = df.groupby("report")["target"].agg(lambda x: x.mode()[0])
        strat_y        = df["report"].map(majority_label).values

        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(sgkf.split(df, strat_y, groups)):
            df.loc[val_idx, "fold"] = fold
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(skf.split(df, df["target"])):
            df.loc[val_idx, "fold"] = fold

    return df


def load_synthetic(classes: list = None, data_dir: str = None) -> pd.DataFrame:
    """Load synthetic mammography reports from the competition's external dataset.

    Args:
        classes: list of int target classes to keep (e.g. [0,3,4,5,6]).
                 None means return all classes.
        data_dir: override data directory; defaults to auto-detected competition dir.

    Returns DataFrame with columns: report, target, text (pre-cleaned).
    """
    if data_dir is None:
        data_dir = find_data_dir()
    synth_path = Path(data_dir) / "synthetic_ext_data" / "mammography_reports_pt_full.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at {synth_path}")
    df = pd.read_csv(synth_path)
    df.columns = df.columns.str.strip()
    df["target"] = df["target"].astype(int)
    if classes is not None:
        df = df[df["target"].isin(classes)].reset_index(drop=True)
    # Use the full report field to match real training data format
    return df[["report", "target"]].copy()


def dedup_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated DataFrame for use as training data.

    Keeps one representative row per unique report text. For groups with
    conflicting labels (11 groups, all with an overwhelming majority), replaces
    the label with the majority vote. The 'fold' column is preserved — group-aware
    folds guarantee all copies of a text share the same fold, so the representative
    row's fold is correct.
    """
    df = df.copy()

    # Fix minority labels in the 11 conflicting groups
    majority_label = df.groupby("report")["target"].agg(lambda x: x.mode()[0])
    df["target"]   = df["report"].map(majority_label)

    dedup = (
        df.drop_duplicates(subset="report")
          .reset_index(drop=False)
          .rename(columns={"index": "orig_idx"})
    )
    return dedup
