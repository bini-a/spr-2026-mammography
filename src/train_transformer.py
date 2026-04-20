"""
GPU-accelerated transformer fine-tuning with OOF cross-validation.
Supports BERT-style models via HuggingFace transformers.

Features:
  - Mixed precision (torch.amp FP16)
  - Multi-GPU via DataParallel (uses all visible GPUs automatically)
  - Gradient accumulation
  - Linear warmup + linear decay LR schedule
  - Class-weighted cross-entropy (handles the 87% class-2 imbalance)
  - Best-epoch checkpointing within each fold
  - Weights & Biases tracking (optional — set wandb.enabled: true in config)
  - Same output structure as train.py (oof_preds.csv, metrics.json, etc.)
"""
import gc
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_linear_schedule_with_warmup
import yaml
from tqdm import tqdm

from src.data import load_train, make_folds
from src.evaluate import append_to_results_log, compute_metrics, print_metrics, save_metrics
from src.features import clean_text
from src.logging_utils import run_log
from src.models.transformer import build_model, save_model
from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]
N_CLASSES = len(CLASSES)


# ── Device setup ─────────────────────────────────────────

def _setup_device(gpu_cfg=None):
    """
    Resolve which GPU(s) to use from the top-level 'gpu' config field.

    Config examples:
      gpu: all      → all available GPUs (DataParallel)
      gpu: 0        → GPU 0 only
      gpu: 1        → GPU 1 only
      gpu: [0, 1]   → both GPUs explicitly

    You can also override at the shell level with CUDA_VISIBLE_DEVICES:
      CUDA_VISIBLE_DEVICES=1 python run.py configs/exp004.yaml --train
    """
    if not torch.cuda.is_available():
        print("No GPU available — running on CPU")
        return torch.device("cpu"), []

    n_available = torch.cuda.device_count()

    if gpu_cfg is None or gpu_cfg == "all":
        gpu_ids = list(range(n_available))
    elif isinstance(gpu_cfg, int):
        gpu_ids = [gpu_cfg]
    elif isinstance(gpu_cfg, list):
        gpu_ids = [int(g) for g in gpu_cfg]
    else:
        raise ValueError(
            f"Invalid 'gpu' config value: {gpu_cfg!r}. "
            "Use an integer, a list of integers, or 'all'."
        )

    for g in gpu_ids:
        if g >= n_available:
            raise ValueError(
                f"GPU {g} requested but only {n_available} GPU(s) available "
                f"(indices 0–{n_available - 1})."
            )

    label = " + ".join(
        f"[{g}] {torch.cuda.get_device_name(g)}" for g in gpu_ids
    )
    print(f"GPUs : {label}")
    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids


# ── Weights & Biases ──────────────────────────────────────

def _init_wandb(config, exp_name):
    """
    Initialise a wandb run and return it, or return None if wandb is
    disabled in config or not installed.
    """
    wb = config.get("wandb", {})
    if not wb.get("enabled", False):
        return None
    try:
        import wandb
        mdl = config["model"]
        run = wandb.init(
            project=wb.get("project", "spr-2026-mammography"),
            entity=wb.get("entity") or None,
            name=exp_name,
            tags=wb.get("tags", []),
            config={
                "experiment":   exp_name,
                "pretrained":   mdl["pretrained"],
                "max_length":   mdl.get("max_length", 512),
                "n_folds":      config["data"]["n_folds"],
                "seed":         config.get("seed", 42),
                **mdl.get("params", {}),
            },
            reinit=True,
        )
        # Custom x-axis so per-fold/epoch charts align correctly
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        print(f"wandb run : {run.url}")
        return run
    except Exception as exc:
        print(f"wandb init failed ({exc}) — continuing without tracking")
        return None


def _wandb_log_epoch(run, fold, epoch, train_loss, ep_metrics, lr):
    if run is None:
        return
    import wandb
    log = {
        "epoch":      epoch,
        "fold":       fold,
        "train/loss": train_loss,
        "val/macro_f1": ep_metrics["macro_f1"],
        "lr":         lr,
    }
    for cls in CLASSES:
        r = ep_metrics["report"].get(str(cls), {})
        log[f"val/f1_c{cls}"]   = r.get("f1-score",  0.0)
        log[f"val/prec_c{cls}"] = r.get("precision", 0.0)
        log[f"val/rec_c{cls}"]  = r.get("recall",    0.0)
    run.log(log)


def _wandb_log_oof(run, oof_metrics, y_true, oof_preds, fold_f1s):
    if run is None:
        return
    import wandb
    # Summary scalars
    run.summary["oof/macro_f1"] = oof_metrics["macro_f1"]
    run.summary["oof/fold_f1_mean"] = float(np.mean(fold_f1s))
    run.summary["oof/fold_f1_std"]  = float(np.std(fold_f1s))
    for cls in CLASSES:
        r = oof_metrics["report"].get(str(cls), {})
        run.summary[f"oof/f1_c{cls}"] = r.get("f1-score", 0.0)

    # Confusion matrix
    run.log({"oof/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true.tolist(),
        preds=oof_preds.tolist(),
        class_names=[f"BI-RADS {c}" for c in CLASSES],
    )})

    # Per-class F1 bar chart
    table = wandb.Table(
        columns=["class", "f1", "precision", "recall", "support"],
        data=[
            [
                f"BI-RADS {cls}",
                oof_metrics["report"].get(str(cls), {}).get("f1-score",  0.0),
                oof_metrics["report"].get(str(cls), {}).get("precision", 0.0),
                oof_metrics["report"].get(str(cls), {}).get("recall",    0.0),
                int(oof_metrics["report"].get(str(cls), {}).get("support", 0)),
            ]
            for cls in CLASSES
        ],
    )
    run.log({"oof/per_class": wandb.plot.bar(table, "class", "f1", title="OOF F1 per class")})


# ── Dataset ───────────────────────────────────────────────

class _TextDataset(Dataset):
    """Tokenizes all texts upfront and caches tensors in memory."""

    def __init__(self, texts, tokenizer, max_length, labels=None):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ── Class weights ─────────────────────────────────────────

def _class_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    total  = counts.sum()
    w = np.zeros(N_CLASSES)
    present = counts > 0
    w[present] = total / (N_CLASSES * counts[present])
    return torch.tensor(w, dtype=torch.float32)


# ── Training helpers ──────────────────────────────────────

def _train_epoch(model, loader, optimizer, scaler, scheduler, criterion, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=scaler.is_enabled()):
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if ttids is not None:
                kwargs["token_type_ids"] = ttids
            out    = model(**kwargs)
            logits = out.logits if hasattr(out, "logits") else out
            loss   = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            params = (model.module if hasattr(model, "module") else model).parameters()
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def _predict(model, loader, device, use_amp):
    model.eval()
    all_probs = []
    for batch in loader:
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if ttids is not None:
                kwargs["token_type_ids"] = ttids
            out    = model(**kwargs)
            logits = out.logits if hasattr(out, "logits") else out

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ── Entry point ───────────────────────────────────────────

def run_training_transformer(config_path: str, notes_override: str = None) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    out_dir  = os.path.join("experiments", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    dest = os.path.join(out_dir, "config.yaml")
    if os.path.abspath(config_path) != os.path.abspath(dest):
        shutil.copy(config_path, dest)

    notes = notes_override or config.get("notes", "")

    with run_log(out_dir):
        _run(config, exp_name, out_dir, config_path, notes)

    return out_dir


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _run(config, exp_name, out_dir, config_path, notes):
    t_start   = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    seed       = config.get("seed", 42)
    n_folds    = config["data"]["n_folds"]
    mdl_cfg    = config["model"]
    pretrained      = mdl_cfg["pretrained"]
    max_length      = mdl_cfg.get("max_length", 512)
    p               = mdl_cfg.get("params", {})
    batch_size      = p.get("batch_size", 32)
    eval_batch_size = p.get("eval_batch_size", batch_size * 2)
    grad_accum      = p.get("gradient_accumulation_steps", 1)
    n_epochs        = p.get("epochs", 5)
    lr              = p.get("learning_rate", 2e-5)
    warmup_ratio    = p.get("warmup_ratio", 0.1)
    weight_decay    = p.get("weight_decay", 0.01)
    fp16            = p.get("fp16", True)

    gpu_cfg = config.get("gpu", "all")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"Experiment : {exp_name}")
    print(f"Config     : {config_path}")
    print(f"Started    : {timestamp}")
    print(f"Git commit : {_git_hash()}")
    print(f"Pretrained : {pretrained}")
    if notes:
        print(f"Notes      : {notes}")
    print(f"{'='*55}")

    device, gpu_ids = _setup_device(gpu_cfg)
    use_amp = fp16 and torch.cuda.is_available()

    wb_run = _init_wandb(config, exp_name)

    # ── Load data ─────────────────────────────────────────
    print("Loading data...", end=" ", flush=True)
    df = load_train()
    df["text"] = df["report"].apply(clean_text)
    df = make_folds(df, n_folds=n_folds, seed=seed)
    print(f"done  ({len(df):,} rows, {n_folds} folds)")

    cw = _class_weights(df["target"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── Tokenize once upfront ────────────────────────────
    print(f"Tokenizing with '{pretrained}'...", end=" ", flush=True)
    _, tokenizer = build_model({"pretrained": pretrained})
    full_ds = _TextDataset(
        df["text"].tolist(), tokenizer, max_length, labels=df["target"].values
    )
    print("done")

    oof_probs = np.zeros((len(df), N_CLASSES))
    fold_f1s  = []
    global_epoch = 0  # monotonically increasing across all folds for wandb x-axis

    # ── OOF cross-validation ──────────────────────────────
    fold_bar = tqdm(range(n_folds), desc="CV folds", unit="fold", ncols=72)
    for fold in fold_bar:
        train_pos = df.index[df["fold"] != fold].tolist()
        val_pos   = df.index[df["fold"] == fold].tolist()

        train_ds = Subset(full_ds, train_pos)
        val_ds   = Subset(full_ds, val_pos)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        model, _ = build_model({"pretrained": pretrained})
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        steps_per_epoch = max(1, len(train_loader) // grad_accum)
        total_steps  = steps_per_epoch * n_epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        base = model.module if hasattr(model, "module") else model
        optimizer = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = GradScaler(enabled=use_amp)

        best_f1    = -1.0
        best_probs = None

        for epoch in range(n_epochs):
            global_epoch += 1
            train_loss = _train_epoch(
                model, train_loader, optimizer, scaler, scheduler,
                criterion, device, grad_accum,
            )
            val_probs  = _predict(model, val_loader, device, use_amp)
            y_val      = df.loc[val_pos, "target"].values
            ep_metrics = compute_metrics(y_val, val_probs.argmax(1))
            ep_f1      = ep_metrics["macro_f1"]
            current_lr = scheduler.get_last_lr()[0]

            fold_bar.set_postfix({
                "fold": f"{fold+1}/{n_folds}",
                "ep":   f"{epoch+1}/{n_epochs}",
                "loss": f"{train_loss:.3f}",
                "F1":   f"{ep_f1:.4f}",
            }, refresh=True)

            _wandb_log_epoch(wb_run, fold + 1, global_epoch, train_loss, ep_metrics, current_lr)

            if ep_f1 > best_f1:
                best_f1    = ep_f1
                best_probs = val_probs.copy()

        for i, pos in enumerate(val_pos):
            oof_probs[pos] = best_probs[i]
        fold_f1s.append(best_f1)

        if wb_run:
            wb_run.log({"fold_best_f1": best_f1, "fold": fold + 1})

        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_bar.close()

    # ── OOF evaluation ────────────────────────────────────
    oof_preds   = oof_probs.argmax(axis=1)
    oof_metrics = compute_metrics(df["target"].values, oof_preds)
    print_metrics(oof_metrics, title=f"OOF Results — {exp_name}")
    print(f"Fold F1s : {[round(f, 4) for f in fold_f1s]}")
    print(f"Mean ± σ : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    oof_df = df[["ID", "target", "fold"]].copy()
    oof_df["oof_pred"] = oof_preds
    for i, cls in enumerate(CLASSES):
        oof_df[f"p{cls}"] = oof_probs[:, i]
    oof_df.to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)
    save_metrics(oof_metrics, out_dir)

    _wandb_log_oof(wb_run, oof_metrics, df["target"].values, oof_preds, fold_f1s)

    # ── Optional threshold tuning ─────────────────────────
    if config.get("threshold", {}).get("enabled", False):
        print("\n--- Threshold Tuning ---")
        offsets = tune_thresholds(oof_probs, df["target"].values, seed=seed)
        np.save(os.path.join(out_dir, "thresholds.npy"), offsets)
        tuned_preds   = oof_probs + offsets
        tuned_metrics = compute_metrics(df["target"].values, tuned_preds.argmax(1))
        print_metrics(tuned_metrics, title="OOF Results (after threshold tuning)")
        save_metrics(tuned_metrics, os.path.join(out_dir, "metrics_tuned.json"))
        oof_metrics = tuned_metrics
        if wb_run:
            wb_run.summary["oof/macro_f1_tuned"] = tuned_metrics["macro_f1"]

    # ── Retrain on full data ──────────────────────────────
    print("\n--- Retraining on full data ---")
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)

    model, tokenizer_full = build_model({"pretrained": pretrained})
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    total_steps  = max(1, len(full_loader) // grad_accum) * n_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    base         = model.module if hasattr(model, "module") else model
    optimizer    = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=use_amp)

    for epoch in tqdm(range(n_epochs), desc="Full retrain", unit="ep", ncols=70):
        _train_epoch(model, full_loader, optimizer, scaler, scheduler,
                     criterion, device, grad_accum)

    save_model(model, tokenizer_full, out_dir)
    print(f"Model saved → experiments/{exp_name}/model/")

    if wb_run:
        wb_run.finish()

    duration = time.time() - t_start
    print(f"\nDuration   : {duration:.0f}s  ({duration/60:.1f}m)")
    append_to_results_log(exp_name, oof_metrics, config,
                          timestamp=timestamp, duration=duration, notes=notes)
    print(f"Outputs    : experiments/{exp_name}/")
