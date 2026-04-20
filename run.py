#!/usr/bin/env python3
"""
Experiment runner — the only entry point you need.

  Train + predict (most common):
    python run.py configs/exp001_tfidf_logreg.yaml

  Train only (no test.csv locally):
    python run.py configs/exp001_tfidf_logreg.yaml --train

  Add a note to the run (overrides config notes field):
    python run.py configs/exp002.yaml --train --notes "trying higher C"

  Predict only (model already trained):
    python run.py configs/exp001_tfidf_logreg.yaml --predict

  Re-run a past experiment from its saved config snapshot:
    python run.py --rerun exp001_tfidf_logreg --train

  Compare all experiments (focused columns):
    python run.py --compare

  Compare with all columns:
    python run.py --compare --full
"""
import argparse
import os
import sys


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _is_transformer(config: dict) -> bool:
    from src.models.transformer import TRANSFORMER_TYPES
    return config.get("model", {}).get("type", "") in TRANSFORMER_TYPES


def cmd_train(config_path: str, notes: str = None):
    config = _load_config(config_path)
    if _is_transformer(config):
        from src.train_transformer import run_training_transformer
        run_training_transformer(config_path, notes_override=notes)
    else:
        from src.train import run_training
        run_training(config_path, notes_override=notes)


def cmd_predict(config_path: str):
    config  = _load_config(config_path)
    exp_dir = os.path.join("experiments", config["experiment_name"])

    if _is_transformer(config):
        model_dir = os.path.join(exp_dir, "model")
        if not os.path.isdir(model_dir):
            print(f"No trained model found in {model_dir}/. Run with --train first.")
            sys.exit(1)
        from src.predict import run_predict_transformer
        run_predict_transformer(exp_dir)
    else:
        if not os.path.exists(os.path.join(exp_dir, "model.pkl")):
            print(f"No trained model found in {exp_dir}/. Run with --train first.")
            sys.exit(1)
        from src.predict import run_predict
        run_predict(exp_dir)


def cmd_notebook(config_path: str):
    config  = _load_config(config_path)
    exp_dir = os.path.join("experiments", config["experiment_name"])
    model_type = config.get("model", {}).get("type", "")

    if model_type in ("transformer", "bert", "xlmr"):
        from src.notebook_gen import generate_transformer_notebook
        if not os.path.isdir(os.path.join(exp_dir, "model")):
            print(f"No trained transformer model found in {exp_dir}/model/. Run --train first.")
            sys.exit(1)
        out = generate_transformer_notebook(exp_dir)
    else:
        from src.notebook_gen import generate_notebook
        if not os.path.exists(os.path.join(exp_dir, "model.pkl")):
            print(f"No trained model found in {exp_dir}/. Run --train first.")
            sys.exit(1)
        out = generate_notebook(exp_dir)

    print(f"Notebook written → {out}")


def cmd_compare(full: bool = False):
    from src.evaluate import print_comparison
    print_comparison(full=full)


def _resolve_config(args) -> str:
    """Return the config path, handling both --rerun and direct config arg."""
    if args.rerun:
        saved = os.path.join("experiments", args.rerun, "config.yaml")
        if not os.path.exists(saved):
            print(f"No saved config found for experiment '{args.rerun}'.")
            print(f"Expected: {saved}")
            sys.exit(1)
        print(f"Re-running from saved config: {saved}")
        return saved
    if not args.config:
        return None
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)
    return args.config


def main():
    parser = argparse.ArgumentParser(
        description="SPR 2026 experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", nargs="?", help="Path to experiment YAML config")
    parser.add_argument("--train",   action="store_true", help="Train only")
    parser.add_argument("--predict", action="store_true", help="Predict only")
    parser.add_argument("--compare", action="store_true", help="Show results leaderboard")
    parser.add_argument("--full",    action="store_true", help="Show all columns in --compare")
    parser.add_argument("--notes",   default=None, metavar="TEXT",
                        help="Freeform note for this run (overrides config notes field)")
    parser.add_argument("--notebook", action="store_true",
                        help="Generate a Kaggle training notebook (requires trained model)")
    parser.add_argument("--notebook-inference", metavar="DATASET_SLUG",
                        help="Generate a fast inference-only notebook; pass Kaggle dataset slug, "
                             "e.g. 'yourusername/exp010-bertimbau-es'")
    parser.add_argument("--rerun",   metavar="EXP_NAME",
                        help="Re-run a past experiment from its saved config snapshot")
    args = parser.parse_args()

    if args.compare:
        cmd_compare(full=args.full)
        return

    config_path = _resolve_config(args)
    if not config_path:
        parser.print_help()
        sys.exit(1)

    if getattr(args, 'notebook_inference', None):
        from src.notebook_gen import generate_inference_notebook
        config  = _load_config(config_path)
        exp_dir = os.path.join("experiments", config["experiment_name"])
        out = generate_inference_notebook(exp_dir, args.notebook_inference)
        print(f"Inference notebook written → {out}")
    elif args.notebook:
        cmd_notebook(config_path)
    elif args.train:
        cmd_train(config_path, notes=args.notes)
    elif args.predict:
        cmd_predict(config_path)
    else:
        # Default: train then predict
        cmd_train(config_path, notes=args.notes)
        try:
            cmd_predict(config_path)
        except FileNotFoundError as e:
            print(f"\n[predict skipped] {e}")


if __name__ == "__main__":
    main()
