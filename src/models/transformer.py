"""
HuggingFace transformer model helpers — build / save / load.
Actual training is handled by src/train_transformer.py.
"""
import os

TRANSFORMER_TYPES = {"transformer", "bert", "xlmr"}
N_CLASSES = 7


def build_model(config: dict):
    """Return (model, tokenizer) initialised from a pretrained checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pretrained = config.get("pretrained", "neuralmind/bert-base-portuguese-cased")
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def save_model(model, tokenizer, out_dir: str) -> None:
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    # Unwrap DataParallel before saving
    m = model.module if hasattr(model, "module") else model
    m.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def load_model(out_dir: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = os.path.join(out_dir, "model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer
