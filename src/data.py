"""
Data loading and fold creation.
Handles both local and Kaggle runtime paths automatically.
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Searched in order — first match wins
_DATA_DIRS = [
    "/kaggle/input/spr-2026-mammography-report-classification",
    "./spr-2026-mammography-report-classification",
    "../spr-2026-mammography-report-classification",
    "../../spr-2026-mammography-report-classification",
]


def find_data_dir() -> str:
    for d in _DATA_DIRS:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError(
        f"Could not find data directory. Tried: {_DATA_DIRS}\n"
        "Place the competition data in ./spr-2026-mammography-report-classification/"
    )


def load_train(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df.columns = df.columns.str.strip()
    df["target"] = df["target"].astype(int)
    return df


def load_test(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"test.csv not found at {path}.\n"
            "This is expected locally — it only exists in the Kaggle evaluation runtime."
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
