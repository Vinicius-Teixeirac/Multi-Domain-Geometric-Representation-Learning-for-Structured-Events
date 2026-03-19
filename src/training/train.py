# src/training/train.py
from pathlib import Path
from typing import Optional, Iterable


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.metrics import compute_classification_metrics
from src.config.paths import ARTIFACTS_DATA


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[Iterable],
    num_epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: Optional[torch.Tensor],
    dataset_name: str,
    patience: int = 30,
    device: str = "cpu",
    metric: str = "f1_macro",
    exp_id: str = "",
):
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    out_dir = ARTIFACTS_DATA / dataset_name / "models" / model.__class__.__name__ / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_model.pt"

    best_metric = -1.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ---------------------
        # Train
        # ---------------------
        model.train()
        train_loss = 0.0
        num_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            logits, targets = model.forward_batch(batch, device)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)


        train_loss /= num_samples

        # ---------------------
        # Validation
        # ---------------------
        val_metric = None
        if val_loader is not None:
            model.eval()
            all_logits, all_targets = [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    logits, targets = model.forward_batch(batch, device)
                    all_logits.append(logits)
                    all_targets.append(targets)

            logits = torch.cat(all_logits).cpu()
            targets = torch.cat(all_targets).cpu()

            metrics = compute_classification_metrics(logits, targets)
            val_metric = metrics[metric]
            
            scheduler.step(val_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "val_metrics": metrics,
                    },
                    best_model_path,
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
                f"Epoch {epoch} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val F1 (macro): {val_metric:.4f} | "
                f"Acc: {metrics['accuracy']:.4f}"
            )

    print(f"Best validation macro-metric: {best_metric:.4f}")
    return best_model_path
