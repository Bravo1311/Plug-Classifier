from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import accuracy_top1


@dataclass
class TrainCfg:
    exp_name: str
    device: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    freeze_backbone_epochs: int
    early_stop_patience: int
    log_every: int
    save_every_epoch: bool

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def _freeze_backbone(model: nn.Module, freeze: bool):
    # Heuristic: freeze everything except the last classifier layer(s)
    for p in model.parameters():
        p.requires_grad = not freeze

    # Unfreeze common classifier heads
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

def _run_epoch(model, loader, optim, criterion, device, train: bool):
    if train:
        """
            Dropout is active
            batchNorm updates running stats
        """
        model.train()
    else:
        """
            Dropout off
            batchNorm uses stored running stats
        """
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        # x.shape = (batch_size, channels, height, width)
        # y = ground truth labels

        x = x.to(device, non_blocking=True) 
        y = y.to(device, non_blocking=True)
        # non_blocking=True speeds up transfer only when DataLoader uses pin_memory=True and source tensor is in pinned memory

        with torch.set_grad_enabled(train):
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optim.step()

        bs = x.size(0)   # batchsize
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)

def train_classifier(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg: TrainCfg,
    out_dir: str,
    class_names,
) -> Dict[str, Any]:
    device = _resolve_device(cfg.device)
    model.to(device)

    run_dir = Path(out_dir) / cfg.exp_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    bad_epochs = 0
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        freeze = epoch <= cfg.freeze_backbone_epochs
        _freeze_backbone(model, freeze=freeze)

        # Rebuild optimizer if freezing changes trainable params (simple approach)
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_loss, train_acc = _run_epoch(model, train_loader, optim, criterion, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, optim, criterion, device, train=False)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        if cfg.save_every_epoch:
            ep_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": list(class_names),
            }, ep_path)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": list(class_names),
            }, best_path)
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} | freeze={freeze}")

        if bad_epochs >= cfg.early_stop_patience:
            print(f"Early stopping: no val improvement for {cfg.early_stop_patience} epochs.")
            break

    writer.close()
    return {"best_val_loss": best_val, "best_checkpoint": str(best_path) if best_path else None}
