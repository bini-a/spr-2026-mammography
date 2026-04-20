"""
Quick smoke test for transformer experiment configs.
Validates: config parsing, model load, tokenization, forward pass, loss, backward pass.
Does NOT run a full training loop — completes in ~2-3 min for all configs.

Usage:
    uv run python scripts/smoke_test_configs.py
    uv run python scripts/smoke_test_configs.py configs/exp006_xlmr_base.yaml  # single config
"""
import os
import sys

# ensure project root is on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

CONFIGS = [
    "configs/exp006_xlmr_base.yaml",
    "configs/exp007_bertimbau_large.yaml",
    "configs/exp008_mdeberta.yaml",
    "configs/exp009_xlmr_large.yaml",
]

# Short PT-BR mammography-style texts, one per class
SAMPLE_TEXTS = [
    "Mama direita sem alterações. Ausência de nódulos, calcificações ou assimetrias.",
    "Nódulo oval de contornos circunscritos no QSE da mama esquerda, estável há 3 anos.",
    "Nódulo de baixa suspeição no quadrante superior externo. Sugere-se controle em 6 meses.",
    "Calcificações pleomórficas agrupadas no QSI da mama direita. Considerar biópsia.",
]
SAMPLE_LABELS = torch.tensor([1, 2, 3, 4], dtype=torch.long)
N_CLASSES = 7


def _focal_loss(logits, labels, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def smoke_test(config_path: str) -> bool:
    t0 = time.time()
    try:
        # ── 1. Parse config ───────────────────────────────
        with open(config_path) as f:
            config = yaml.safe_load(f)

        exp_name   = config["experiment_name"]
        pretrained = config["model"]["pretrained"]
        max_length = config["model"].get("max_length", 512)
        p          = config["model"].get("params", {})
        loss_fn    = p.get("loss_fn", "cross_entropy")
        focal_gamma = p.get("focal_gamma", 2.0)
        fp16       = p.get("fp16", True)

        print(f"\n{'─'*60}")
        print(f"  {exp_name}")
        print(f"  model      : {pretrained}")
        print(f"  loss_fn    : {loss_fn}" + (f"  γ={focal_gamma}" if loss_fn == "focal" else ""))
        print(f"  fp16       : {fp16}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = fp16 and torch.cuda.is_available()

        # ── 2. Load model + tokenizer ─────────────────────
        print("  [1/5] loading model & tokenizer ...", end=" ", flush=True)
        from src.models.transformer import build_model
        model, tokenizer = build_model({"pretrained": pretrained})
        model = model.to(device)
        print(f"done  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

        # ── 3. Tokenize ───────────────────────────────────
        print("  [2/5] tokenizing sample batch ...", end=" ", flush=True)
        enc = tokenizer(
            SAMPLE_TEXTS,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        ttids = enc.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device)
        labels = SAMPLE_LABELS.to(device)
        print(f"done  (seq_len={ids.shape[1]})")

        # ── 4. Forward pass ───────────────────────────────
        print("  [3/5] forward pass ...", end=" ", flush=True)
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(enabled=use_amp)
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if ttids is not None:
            kwargs["token_type_ids"] = ttids

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
            out    = model(**kwargs)
            logits = out.logits if hasattr(out, "logits") else out
        assert logits.shape == (len(SAMPLE_TEXTS), N_CLASSES), \
            f"unexpected logits shape {logits.shape}, expected ({len(SAMPLE_TEXTS)}, {N_CLASSES})"
        print(f"done  (logits {logits.shape})")

        # ── 5. Loss computation ───────────────────────────
        print("  [4/5] loss computation ...", end=" ", flush=True)
        counts = torch.bincount(SAMPLE_LABELS, minlength=N_CLASSES).float()
        cw = torch.where(counts > 0, counts.sum() / (N_CLASSES * counts), torch.zeros_like(counts))
        cw = cw.to(device)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
            if loss_fn == "focal":
                loss = _focal_loss(logits.float(), labels, gamma=focal_gamma, weight=cw)
            else:
                loss = nn.CrossEntropyLoss(weight=cw)(logits.float(), labels)
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        print(f"done  (loss={loss.item():.4f})")

        # ── 6. Backward pass ──────────────────────────────
        print("  [5/5] backward pass ...", end=" ", flush=True)
        model.zero_grad()
        scaler.scale(loss).backward()
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert grad_norm > 0, "grad norm is zero — no gradients flowed"
        print(f"done  (grad_norm={grad_norm:.3f})")

        elapsed = time.time() - t0
        print(f"  PASS  ({elapsed:.1f}s)")
        return True

    except Exception:
        elapsed = time.time() - t0
        print(f"\n  FAIL  ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


def main():
    configs = sys.argv[1:] if len(sys.argv) > 1 else CONFIGS

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — tests will run on CPU (slower, no FP16)")

    results = {}
    for cfg in configs:
        results[cfg] = smoke_test(cfg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for cfg, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {cfg}")
        if not passed:
            all_pass = False

    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
