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
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import yaml
from tqdm import tqdm

from src.data import dedup_for_training, load_synthetic, load_train, make_folds
from src.evaluate import append_to_results_log, compute_metrics, print_metrics, save_metrics
from src.features import clean_text
from src.logging_utils import run_log
from src.models.transformer import build_model, save_model
from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]
N_CLASSES = len(CLASSES)


class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets, reduction: str = "mean"):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if reduction == "none":
            return loss
        return loss.mean()


def _ce_loss(logits, labels, criterion, sample_weight=None):
    """Compute CE (or focal) loss, optionally with per-sample weights."""
    if sample_weight is None:
        return criterion(logits, labels)
    if isinstance(criterion, _FocalLoss):
        per_example = criterion(logits, labels, reduction="none")
    else:
        per_example = F.cross_entropy(
            logits, labels,
            weight=criterion.weight,
            reduction="none",
            label_smoothing=getattr(criterion, "label_smoothing", 0.0),
        )
    return (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


class _AWP:
    """
    Adversarial Weight Perturbation.
    After the first forward-backward pass has accumulated gradients, call
    attack_backward() to:
      1. perturb model weights in the gradient direction (bounded by adv_eps)
      2. run a second forward-backward to accumulate adversarial gradients
      3. restore original weights
    The optimizer then steps with the sum of both gradient contributions.
    """

    def __init__(self, model, adv_lr: float = 1e-4, adv_eps: float = 1e-2,
                 start_epoch: int = 1):
        self.model      = model
        self.adv_lr     = adv_lr
        self.adv_eps    = adv_eps
        self.start_epoch = start_epoch
        self._backup    = {}
        self._grad_dir  = {}

    def attack_backward(self, kwargs, labels, criterion, sample_weight, scaler, use_amp):
        self._save()
        self._attack()
        with autocast(device_type="cuda", enabled=use_amp):
            out_adv = self.model(**kwargs)
            logits_adv = (out_adv.logits if hasattr(out_adv, "logits") else out_adv).float()
            loss_adv = _ce_loss(logits_adv, labels, criterion, sample_weight)
        scaler.scale(loss_adv).backward()
        self._restore()

    def _save(self):
        self._backup   = {}
        self._grad_dir = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._backup[name] = param.data.clone()
                g_norm = torch.norm(param.grad)
                if g_norm > 0 and not torch.isnan(g_norm):
                    self._grad_dir[name] = param.grad.detach() / g_norm

    def _attack(self):
        for name, param in self.model.named_parameters():
            if name in self._grad_dir:
                param.data.add_(self.adv_lr * self._grad_dir[name])
                # Clip perturbation magnitude
                diff = param.data - self._backup[name]
                diff.clamp_(-self.adv_eps, self.adv_eps)
                param.data.copy_(self._backup[name] + diff)

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup   = {}
        self._grad_dir = {}


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

    def __init__(self, texts, tokenizer, max_length, labels=None, sample_weights=None):
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
        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None else None
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
        if self.sample_weights is not None:
            item["sample_weight"] = self.sample_weights[idx]
        return item


# ── Class weights ─────────────────────────────────────────

def _class_weights(y, sample_weights=None, num_classes=None):
    num_classes = int(num_classes or N_CLASSES)
    if sample_weights is None:
        counts = np.bincount(y, minlength=num_classes).astype(float)
    else:
        counts = np.bincount(
            y, weights=np.asarray(sample_weights, dtype=float), minlength=num_classes
        ).astype(float)
    total  = counts.sum()
    w = np.zeros(num_classes)
    present = counts > 0
    w[present] = total / (num_classes * counts[present])
    return torch.tensor(w, dtype=torch.float32)


# ── Optimizer ────────────────────────────────────────────

def _make_sampler(dataset_labels: np.ndarray, num_classes: int | None = None) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler where each sample's weight is 1/class_count."""
    num_classes = int(num_classes or (dataset_labels.max() + 1))
    counts = np.bincount(dataset_labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)  # avoid division by zero for absent classes
    weights = 1.0 / counts[dataset_labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(dataset_labels),
        replacement=True,
    )


def _build_optimizer(base_model, lr, weight_decay, llrd_decay=1.0):
    """
    Build AdamW. When llrd_decay < 1.0, applies layer-wise LR decay so lower
    BERT layers (which encode pretrained knowledge) are updated more slowly
    than the classification head. Works for any HuggingFace BERT-family model.

    LR assignment:
      embeddings  : lr * decay^(n_layers)        ← smallest
      layer k     : lr * decay^(n_layers - 1 - k)
      head/pooler : lr                            ← full LR
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    if llrd_decay >= 1.0:
        # Standard flat-LR AdamW
        decay_params   = [p for n, p in base_model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)]
        nodecay_params = [p for n, p in base_model.named_parameters()
                          if p.requires_grad and     any(nd in n for nd in no_decay)]
        return torch.optim.AdamW(
            [{"params": decay_params,   "lr": lr, "weight_decay": weight_decay},
             {"params": nodecay_params, "lr": lr, "weight_decay": 0.0}]
        )

    # Detect number of transformer layers from parameter names
    layer_indices = set()
    for name, _ in base_model.named_parameters():
        m = re.search(r"\.layer\.(\d+)\.", name)
        if m:
            layer_indices.add(int(m.group(1)))
    n_layers = max(layer_indices) + 1 if layer_indices else 1

    param_groups = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        m = re.search(r"\.layer\.(\d+)\.", name)
        if m:
            k = int(m.group(1))
            group_lr = lr * (llrd_decay ** (n_layers - 1 - k))
        elif "embeddings" in name:
            group_lr = lr * (llrd_decay ** n_layers)
        else:
            group_lr = lr   # head, pooler, classifier
        param_groups.append({"params": [param], "lr": group_lr, "weight_decay": wd})

    return torch.optim.AdamW(param_groups)


# ── Training helpers ──────────────────────────────────────

def _train_epoch(model, loader, optimizer, scaler, scheduler, criterion, device, grad_accum,
                 rdrop_alpha: float = 0.0, awp: "_AWP | None" = None, epoch: int = 0):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_amp = scaler.is_enabled()

    for step, batch in enumerate(loader):
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(device, non_blocking=True)

        kwargs = {"input_ids": ids, "attention_mask": mask}
        if ttids is not None:
            kwargs["token_type_ids"] = ttids

        with autocast(device_type="cuda", enabled=use_amp):
            out1    = model(**kwargs)
            logits1 = (out1.logits if hasattr(out1, "logits") else out1).float()

            if rdrop_alpha > 0.0:
                # R-Drop: second forward pass with different dropout mask
                out2    = model(**kwargs)
                logits2 = (out2.logits if hasattr(out2, "logits") else out2).float()
                ce_loss = 0.5 * (
                    _ce_loss(logits1, labels, criterion, sample_weight) +
                    _ce_loss(logits2, labels, criterion, sample_weight)
                )
                p1 = F.log_softmax(logits1, dim=-1)
                p2 = F.log_softmax(logits2, dim=-1)
                kl = 0.5 * (
                    F.kl_div(p1, p2.exp().detach(), reduction="batchmean") +
                    F.kl_div(p2, p1.exp().detach(), reduction="batchmean")
                )
                loss = ce_loss + rdrop_alpha * kl
            else:
                loss = _ce_loss(logits1, labels, criterion, sample_weight)

            loss = loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if awp is not None and epoch >= awp.start_epoch:
                awp.attack_backward(kwargs, labels, criterion, sample_weight, scaler, use_amp)

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


def _specialist_class_maps(classes):
    classes = [int(c) for c in classes]
    local_to_global = {i: cls for i, cls in enumerate(classes)}
    global_to_local = {cls: i for i, cls in local_to_global.items()}
    return global_to_local, local_to_global


def _expand_probs_to_global(local_probs: np.ndarray, local_to_global: dict[int, int]) -> np.ndarray:
    global_probs = np.zeros((len(local_probs), N_CLASSES), dtype=float)
    for local_idx, global_cls in local_to_global.items():
        global_probs[:, global_cls] = local_probs[:, local_idx]
    return global_probs


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
    loss_fn                  = p.get("loss_fn", "cross_entropy")   # "cross_entropy" | "focal"
    focal_gamma              = p.get("focal_gamma", 2.0)
    early_stopping_patience  = p.get("early_stopping_patience", None)  # None = disabled
    label_smoothing          = p.get("label_smoothing", 0.0)           # 0.0 = disabled
    llrd_decay               = p.get("llrd_decay", 1.0)               # 1.0 = disabled (flat LR)
    weighted_sampler         = p.get("weighted_sampler", False)        # oversample rare classes per batch
    rdrop_alpha              = float(p.get("rdrop_alpha", 0.0))        # 0.0 = disabled; ~1.0 = R-Drop
    awp_cfg                  = config.get("awp", {})
    use_awp                  = awp_cfg.get("enabled", False)

    synth_cfg     = config.get("synthetic_augment", {})
    use_synthetic = synth_cfg.get("enabled", False)
    synth_classes = synth_cfg.get("classes", None)  # None = all classes
    synth_loss_weight = float(synth_cfg.get("loss_weight", 1.0))
    subset_cfg     = config.get("label_subset", {})
    use_label_subset = subset_cfg.get("enabled", False)
    subset_classes = [int(c) for c in subset_cfg.get("classes", [])]
    if use_label_subset and not subset_classes:
        raise ValueError("label_subset.enabled=true requires label_subset.classes")
    if use_label_subset and use_synthetic:
        raise ValueError("synthetic_augment is disabled for label_subset specialist runs in v1")

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
    if llrd_decay < 1.0:
        print(f"LLRD decay : {llrd_decay}")
    if rdrop_alpha > 0.0:
        print(f"R-Drop α   : {rdrop_alpha}")
    if use_awp:
        print(f"AWP        : adv_lr={awp_cfg.get('adv_lr', 1e-4)}  "
              f"adv_eps={awp_cfg.get('adv_eps', 1e-2)}  "
              f"start_epoch={awp_cfg.get('start_epoch', 1)}")
    if use_label_subset:
        print(f"Label subset: {subset_classes}")
    print(f"{'='*55}")

    device, gpu_ids = _setup_device(gpu_cfg)
    use_amp = fp16 and torch.cuda.is_available()

    wb_run = _init_wandb(config, exp_name)

    # ── Load data ─────────────────────────────────────────
    print("Loading data...", end=" ", flush=True)
    df = load_train()
    df["text"] = df["report"].apply(clean_text)
    df = make_folds(df, n_folds=n_folds, seed=seed, group_aware=True)
    full_df = df.copy()

    if use_label_subset:
        global_to_local, local_to_global = _specialist_class_maps(subset_classes)
        with open(os.path.join(out_dir, "class_map.json"), "w") as f:
            json.dump(
                {
                    "global_to_local": {str(k): v for k, v in global_to_local.items()},
                    "local_to_global": {str(k): v for k, v in local_to_global.items()},
                },
                f,
                indent=2,
            )
        df_train_src = full_df[full_df["target"].isin(subset_classes)].copy()
        df_train = dedup_for_training(df_train_src)
        df_train["target_local"] = df_train["target"].map(global_to_local).astype(int)
        full_df["target_local"] = full_df["target"].map(global_to_local)
        metric_labels = subset_classes
        model_num_labels = len(subset_classes)
    else:
        df_train = dedup_for_training(full_df)   # one row per unique text, majority-vote labels
        metric_labels = CLASSES
        model_num_labels = N_CLASSES

    # Optional synthetic augmentation — always goes into the training pool,
    # never into OOF validation (which uses `full_ds` = all 18k real rows).
    df_synth = None
    n_synth  = 0
    if use_synthetic:
        df_synth = load_synthetic(classes=synth_classes)
        df_synth["text"] = df_synth["report"].apply(clean_text)
        n_synth = len(df_synth)

    print(
        f"done  ({len(full_df):,} real rows → {len(df_train):,} unique real texts"
        + (f" + {n_synth:,} synthetic" if use_synthetic else "")
        + f" for training, {n_folds} folds)"
    )
    if use_synthetic:
        synth_label = synth_classes if synth_classes is not None else "all"
        print(f"Synthetic classes: {synth_label}")
        print(f"Synthetic loss weight: {synth_loss_weight:g}")

    # Class weights from combined training distribution (real + synthetic if enabled)
    train_labels_for_cw = (
        df_train["target_local"].values if use_label_subset else df_train["target"].values
    )
    train_sample_weights_for_cw = np.ones(len(df_train), dtype=float)
    if df_synth is not None:
        train_labels_for_cw = np.concatenate([train_labels_for_cw, df_synth["target"].values])
        train_sample_weights_for_cw = np.concatenate([
            train_sample_weights_for_cw,
            np.full(len(df_synth), synth_loss_weight, dtype=float),
        ])
    cw = _class_weights(
        train_labels_for_cw,
        train_sample_weights_for_cw,
        num_classes=model_num_labels,
    ).to(device)
    if loss_fn == "focal":
        criterion = _FocalLoss(gamma=focal_gamma, weight=cw)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # ── Tokenize once upfront ────────────────────────────
    print(f"Tokenizing with '{pretrained}'...", end=" ", flush=True)
    _, tokenizer = build_model({"pretrained": pretrained, "num_labels": model_num_labels})
    # full_ds: all real rows — used for validation. For specialists, labels are
    # not consumed from the dataset because metrics are computed from full_df.
    full_ds = _TextDataset(
        full_df["text"].tolist(),
        tokenizer,
        max_length,
        labels=np.zeros(len(full_df), dtype=int),
    )
    # train_ds_base: real unique texts for training.
    # If synthetic augmentation is enabled, aug_ds extends this with synthetic rows;
    # synthetic items occupy positions [len(df_train), len(df_train)+n_synth).
    if df_synth is not None:
        aug_texts  = df_train["text"].tolist() + df_synth["text"].tolist()
        real_train_labels = (
            df_train["target_local"].values if use_label_subset else df_train["target"].values
        )
        aug_labels = np.concatenate([real_train_labels, df_synth["target"].values])
        aug_sample_weights = np.concatenate([
            np.ones(len(df_train), dtype=float),
            np.full(len(df_synth), synth_loss_weight, dtype=float),
        ])
        aug_ds = _TextDataset(
            aug_texts,
            tokenizer,
            max_length,
            labels=aug_labels,
            sample_weights=aug_sample_weights,
        )
        dedup_ds = aug_ds   # alias for retrain loop
        synth_positions = list(range(len(df_train), len(df_train) + n_synth))
    else:
        dedup_ds = _TextDataset(
            df_train["text"].tolist(),
            tokenizer,
            max_length,
            labels=(df_train["target_local"].values if use_label_subset else df_train["target"].values),
        )
        synth_positions = []
    print("done")

    oof_probs        = np.zeros((len(full_df), N_CLASSES))
    fold_f1s         = []
    fold_best_epochs = []
    global_epoch     = 0  # monotonically increasing across all folds for wandb x-axis

    # ── OOF cross-validation ──────────────────────────────
    fold_bar = tqdm(range(n_folds), desc="CV folds", unit="fold", ncols=72)
    for fold in fold_bar:
        # Train on deduplicated real rows not in this fold + all synthetic rows;
        # validate on all real rows in this fold (honest OOF — no synthetic in val).
        real_train_pos = df_train.index[df_train["fold"] != fold].tolist()
        train_pos = real_train_pos + synth_positions
        val_pos   = full_df.index[full_df["fold"] == fold].tolist()

        train_ds = Subset(dedup_ds, train_pos)
        val_ds   = Subset(full_ds, val_pos)

        if weighted_sampler:
            real_labels = (
                df_train.loc[df_train["fold"] != fold, "target_local"].values
                if use_label_subset else
                df_train.loc[df_train["fold"] != fold, "target"].values
            )
            fold_labels = (
                np.concatenate([real_labels, df_synth["target"].values])
                if df_synth is not None else real_labels
            )
            sampler = _make_sampler(fold_labels, num_classes=model_num_labels)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        model, _ = build_model({"pretrained": pretrained, "num_labels": model_num_labels})
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        steps_per_epoch = max(1, len(train_loader) // grad_accum)
        total_steps  = steps_per_epoch * n_epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        base = model.module if hasattr(model, "module") else model
        optimizer = _build_optimizer(base, lr, weight_decay, llrd_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = GradScaler(enabled=use_amp)

        awp = None
        if use_awp:
            base_for_awp = model.module if hasattr(model, "module") else model
            awp = _AWP(
                base_for_awp,
                adv_lr=float(awp_cfg.get("adv_lr", 1e-4)),
                adv_eps=float(awp_cfg.get("adv_eps", 1e-2)),
                start_epoch=int(awp_cfg.get("start_epoch", 1)),
            )

        best_f1    = -1.0
        best_probs = None
        best_epoch = 0
        no_improve = 0

        for epoch in range(n_epochs):
            global_epoch += 1
            train_loss = _train_epoch(
                model, train_loader, optimizer, scaler, scheduler,
                criterion, device, grad_accum,
                rdrop_alpha=rdrop_alpha, awp=awp, epoch=epoch,
            )
            val_probs_local  = _predict(model, val_loader, device, use_amp)
            val_probs = (
                _expand_probs_to_global(val_probs_local, local_to_global)
                if use_label_subset else val_probs_local
            )
            val_mask = (
                full_df.loc[val_pos, "target"].isin(subset_classes).values
                if use_label_subset else np.ones(len(val_pos), dtype=bool)
            )
            y_val = full_df.loc[val_pos, "target"].values[val_mask]
            ep_metrics = compute_metrics(
                y_val,
                val_probs[val_mask].argmax(1),
                labels=metric_labels,
            )
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
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if early_stopping_patience is not None and no_improve >= early_stopping_patience:
                    tqdm.write(f"  early stop fold {fold+1} at epoch {epoch+1} "
                               f"(no improvement for {early_stopping_patience} epochs)")
                    break

        for i, pos in enumerate(val_pos):
            oof_probs[pos] = best_probs[i]
        fold_f1s.append(best_f1)
        fold_best_epochs.append(best_epoch)

        if wb_run:
            wb_run.log({"fold_best_f1": best_f1, "fold": fold + 1})

        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_bar.close()

    # ── OOF evaluation ────────────────────────────────────
    eval_mask = (
        full_df["target"].isin(subset_classes).values
        if use_label_subset else np.ones(len(full_df), dtype=bool)
    )
    oof_preds   = oof_probs.argmax(axis=1)
    oof_metrics = compute_metrics(full_df["target"].values[eval_mask], oof_preds[eval_mask], labels=metric_labels)
    print_metrics(oof_metrics, title=f"OOF Results — {exp_name}", labels=metric_labels)
    print(f"Fold F1s : {[round(f, 4) for f in fold_f1s]}")
    print(f"Mean ± σ : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    oof_df = full_df[["ID", "target", "fold"]].copy()
    oof_df["oof_pred"] = oof_preds
    for i, cls in enumerate(CLASSES):
        oof_df[f"p{cls}"] = oof_probs[:, i]
    oof_df.to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)
    save_metrics(oof_metrics, out_dir)

    _wandb_log_oof(wb_run, oof_metrics, full_df["target"].values[eval_mask], oof_preds[eval_mask], fold_f1s)

    # ── Optional threshold tuning ─────────────────────────
    if config.get("threshold", {}).get("enabled", False):
        print("\n--- Threshold Tuning ---")
        if use_label_subset:
            raise ValueError("Threshold tuning is disabled for label_subset specialist runs in v1")
        offsets = tune_thresholds(oof_probs, full_df["target"].values, seed=seed)
        np.save(os.path.join(out_dir, "thresholds.npy"), offsets)
        tuned_preds   = oof_probs + offsets
        tuned_metrics = compute_metrics(full_df["target"].values, tuned_preds.argmax(1))
        print_metrics(tuned_metrics, title="OOF Results (after threshold tuning)")
        save_metrics(tuned_metrics, os.path.join(out_dir, "metrics_tuned.json"))
        oof_metrics = tuned_metrics
        if wb_run:
            wb_run.summary["oof/macro_f1_tuned"] = tuned_metrics["macro_f1"]

    # ── Retrain on full data ──────────────────────────────
    # Use mean best epoch from OOF (not fixed n_epochs) so the submission
    # model is trained for the same number of epochs that OOF validated.
    retrain_epochs = max(1, round(sum(fold_best_epochs) / n_folds))
    print(f"\n--- Retraining on full data ({retrain_epochs} epochs, "
          f"mean of OOF best epochs {fold_best_epochs}) ---")
    if weighted_sampler:
        full_labels = train_labels_for_cw   # already includes synthetic if enabled
        full_sampler = _make_sampler(full_labels, num_classes=model_num_labels)
        full_loader = DataLoader(dedup_ds, batch_size=batch_size, sampler=full_sampler,
                                 num_workers=4, pin_memory=True)
    else:
        full_loader = DataLoader(dedup_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)

    model, tokenizer_full = build_model({"pretrained": pretrained, "num_labels": model_num_labels})
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    total_steps  = max(1, len(full_loader) // grad_accum) * retrain_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    base         = model.module if hasattr(model, "module") else model
    optimizer    = _build_optimizer(base, lr, weight_decay, llrd_decay)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=use_amp)

    retrain_awp = None
    if use_awp:
        retrain_base = model.module if hasattr(model, "module") else model
        retrain_awp = _AWP(
            retrain_base,
            adv_lr=float(awp_cfg.get("adv_lr", 1e-4)),
            adv_eps=float(awp_cfg.get("adv_eps", 1e-2)),
            start_epoch=int(awp_cfg.get("start_epoch", 1)),
        )

    for epoch in tqdm(range(retrain_epochs), desc="Full retrain", unit="ep", ncols=70):
        _train_epoch(model, full_loader, optimizer, scaler, scheduler,
                     criterion, device, grad_accum,
                     rdrop_alpha=rdrop_alpha, awp=retrain_awp, epoch=epoch)

    save_model(model, tokenizer_full, out_dir)
    print(f"Model saved → experiments/{exp_name}/model/")

    if wb_run:
        wb_run.finish()

    duration = time.time() - t_start
    print(f"\nDuration   : {duration:.0f}s  ({duration/60:.1f}m)")
    if use_label_subset:
        print("Specialist label-subset run — skipping append_to_results_log.")
    else:
        append_to_results_log(exp_name, oof_metrics, config,
                              timestamp=timestamp, duration=duration, notes=notes)
    print(f"Outputs    : experiments/{exp_name}/")
