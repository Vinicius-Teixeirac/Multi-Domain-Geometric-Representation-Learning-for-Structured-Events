# src/training/train.py
"""
Central training-loop module.

Provides ``train_model``, the shared fit routine invoked by every model
family runner (MLP, GNN, BERT, multi-domain geometric model). Handles
the optimization loop, learning-rate scheduling on a plateau, early
stopping on a monitored validation metric, and checkpointing of the
best model to ARTIFACTS_DATA.
"""

from pathlib import Path
from typing import Optional, Iterable


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.metrics import compute_classification_metrics
from src.utils.experiments_logging import get_logger
from src.config.paths import ARTIFACTS_DATA

logger = get_logger(__name__)


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
    lr_patience: int | None = None,
    device: str = "cpu",
    metric: str = "f1_macro",
    exp_id: str = "",
) -> Path:
    """Train model with early stopping; save and return the best checkpoint path.

    Validation metrics are tracked each epoch; training halts when the monitored
    metric does not improve for `patience` consecutive epochs.  When no
    val_loader is provided the final epoch weights are saved instead.

    Args:
        model: Model to train; must implement ``forward_batch(batch, device)``
            returning ``(logits, targets)``.
        train_loader: DataLoader yielding training batches.
        val_loader: DataLoader (or other iterable) yielding validation
            batches, or None to skip validation/early stopping and save
            the final-epoch weights instead.
        num_epochs: Maximum number of training epochs.
        lr: Initial learning rate for the Adam optimizer.
        weight_decay: L2 weight decay passed to the Adam optimizer.
        class_weights: Optional per-class weights for CrossEntropyLoss,
            e.g. to counter class imbalance.
        dataset_name: Dataset name, used to build the checkpoint output
            directory under ARTIFACTS_DATA.
        patience: Number of consecutive non-improving validation epochs
            before early stopping triggers. Default 30.
        lr_patience: Number of non-improving epochs before
            ReduceLROnPlateau halves the learning rate. If None, derived
            from `patience` (see inline comment).
        device: Torch device string ("cpu" or "cuda...").
        metric: Key into the dict returned by
            ``compute_classification_metrics`` used to select the best
            checkpoint (e.g. "f1_macro").
        exp_id: Experiment identifier, used to namespace the checkpoint
            output directory under ARTIFACTS_DATA.

    Returns:
        Path to the saved best-checkpoint file (or final-epoch checkpoint
        when no validation loader is provided).
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR patience is deliberately tighter than early-stop patience (1/5th of
    # it) so the learning rate drops and gives training a chance to recover
    # before early stopping gives up entirely.
    effective_lr_patience = lr_patience if lr_patience is not None else max(1, patience // 5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=effective_lr_patience, factor=0.5  # halve LR on plateau (standard convention)
    )

    out_dir = ARTIFACTS_DATA / dataset_name / "models" / model.__class__.__name__ / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_model.pt"

    best_metric = -1.0  # sentinel: any real metric value (0..1) will be an improvement
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
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if val_loader is not None:
            logger.info(
                f"Epoch {epoch} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val {metric}: {val_metric:.4f} | "
                f"Acc: {metrics['accuracy']:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch} | Train loss: {train_loss:.4f}")

    # When there is no validation set, save the final model so evaluate_model can load it.
    if val_loader is None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": num_epochs,
                "best_metric": -1.0,
                "val_metrics": {},
            },
            best_model_path,
        )

    logger.info(f"Best validation macro-metric: {best_metric:.4f}")
    return best_model_path
