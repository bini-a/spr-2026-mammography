"""
Sparse linear models: Logistic Regression and LinearSVC (calibrated).
These are the recommended first baseline — fast, interpretable, strong for text.
"""
import os
import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_model(config: dict):
    """
    Build a model from the 'model' section of an experiment config.
    Supported types: logistic_regression, linear_svc
    Both return a sklearn estimator with predict_proba support.
    """
    model_type = config.get("type", "logistic_regression")
    params = config.get("params", {})
    seed = params.get("seed", 42)

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight=params.get("class_weight", "balanced"),
            solver=params.get("solver", "lbfgs"),
            random_state=seed,
            n_jobs=-1,
        )

    if model_type == "linear_svc":
        # LinearSVC has no predict_proba; wrap with isotonic calibration
        base = LinearSVC(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight=params.get("class_weight", "balanced"),
            random_state=seed,
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=3)

    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        "Choose from: logistic_regression, linear_svc"
    )


def save_model(model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model(out_dir: str):
    with open(os.path.join(out_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)
