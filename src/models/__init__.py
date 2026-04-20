"""
Model dispatcher — routes build/save/load to the right backend by model type.
Add new model families here; train.py and predict.py import only from src.models.
"""
import os
import pickle

LINEAR_TYPES = {"logistic_regression", "linear_svc"}
GBM_TYPES = {"lgbm"}


def build_model(config: dict):
    model_type = config.get("type", "logistic_regression")
    if model_type in LINEAR_TYPES:
        from src.models.linear import build_model as _build
        return _build(config)
    if model_type in GBM_TYPES:
        from src.models.gbm import build_model as _build
        return _build(config)
    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        f"Choose from: {sorted(LINEAR_TYPES | GBM_TYPES)}"
    )


def save_model(model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model(out_dir: str):
    with open(os.path.join(out_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)
