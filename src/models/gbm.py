"""
Gradient boosting models. Currently supports LightGBM (lgbm).
Handles sparse TF-IDF input natively — no dense conversion needed.
"""
import os
import pickle


def build_model(config: dict):
    """
    Build a LightGBM classifier from the 'model' section of an experiment config.
    Supported types: lgbm
    """
    from lightgbm import LGBMClassifier

    model_type = config.get("type", "lgbm")
    params = config.get("params", {})
    seed = params.get("seed", 42)

    if model_type == "lgbm":
        # Use is_unbalance (LightGBM-native) rather than sklearn's class_weight='balanced'.
        # sklearn's balanced mode creates 500:1 sample-weight ratios on this dataset
        # which destabilises gradient updates. is_unbalance adjusts class priors gently.
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 500),
            num_leaves=params.get("num_leaves", 63),
            learning_rate=params.get("learning_rate", 0.1),
            feature_fraction=params.get("feature_fraction", 0.3),
            bagging_fraction=params.get("bagging_fraction", 0.8),
            bagging_freq=params.get("bagging_freq", 1),
            min_child_samples=params.get("min_child_samples", 5),
            is_unbalance=params.get("is_unbalance", True),
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )

    raise ValueError(f"Unknown GBM model type: '{model_type}'. Choose from: lgbm")


def save_model(model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model(out_dir: str):
    with open(os.path.join(out_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)
