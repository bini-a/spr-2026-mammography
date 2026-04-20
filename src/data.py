"""
Data loading and fold creation.
Handles both local and Kaggle runtime paths automatically.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

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


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a 'fold' column (0..n_folds-1) using stratified split on target."""
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["target"])):
        df.loc[val_idx, "fold"] = fold
    return df
