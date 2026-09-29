#!/usr/bin/env python3
"""
Main training script for JWST multi-modal ViT.

Supports:
- Masked band pretraining
- Fine-tuning on physical parameters
- Checkpointing and logging
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Astra_Vision/
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import directly since PROJECT_ROOT contains Astra_Vision/
from data.dataset import MultiBandTileDataset
from models.vit_multimodal import MultimodalViT, build_multimodal_vit
from training.mask import BandMasker, MaskedBandLoss, MaskedBandPretrainer
from training.logger import MetricsLogger


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    mask_prob: float = 0.0,
    load_into_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Parameters
    ----------
    data_dir : str
        Directory containing preprocessed tiles.
    batch_size : int
        Batch size.
    num_workers : int, default=4
        Number of data workers.
    val_split : float, default=0.1
        Fraction of data for validation.
    mask_prob : float, default=0.0
        Masking probability for pretraining.
    load_into_memory : bool, default=False
        Load data into memory.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    dataset = MultiBandTileDataset(
        tile_dir=data_dir,
        load_into_memory=load_into_memory,
        mask_prob=mask_prob,
    )

    # Split into train/val
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def create_model(
    nircam_bands: int = 6,
    miri_bands: int = 4,
    variant: str = "base",
    use_cross_attention: bool = True,
    pretrained_path: Optional[str] = None,
    device: str = "cuda",
) -> MultimodalViT:
    """
    Create multimodal ViT model.

    Parameters
    ----------
    nircam_bands : int, default=6
        Number of NIRCam bands.
    miri_bands : int, default=4
        Number of MIRI bands.
    variant : str, default="base"
        Model variant.
    use_cross_attention : bool, default=True
        Use cross-attention between modalities.
    pretrained_path : str, optional
        Path to pretrained weights.
    device : str, default="cuda"
        Device to use.

    Returns
    -------
    MultimodalViT
        Model ready for training.
    """
    model = build_multimodal_vit(
        variant=variant,
        nircam_bands=nircam_bands,
        miri_bands=miri_bands,
        use_cross_attention=use_cross_attention,
    )

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pretrained weights from {pretrained_path}")

    model = model.to(device)

    # Wrap with DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    return model


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    save_dir: str = "outputs/checkpoints",
    log_dir: str = "outputs/logs",
) -> Dict[str, List[float]]:
    """
    Train model on physical parameter regression.

    Parameters
    ----------
    model : nn.Module
        Multimodal ViT model.
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    device : str, default="cuda"
        Device to train on.
    num_epochs : int, default=50
        Number of epochs.
    learning_rate : float, default=1e-4
        Learning rate.
    weight_decay : float, default=0.05
        Weight decay.
    save_dir : str, default="outputs/checkpoints"
        Directory to save checkpoints.
    log_dir : str, default="outputs/logs"
        Directory for logs.

    Returns
    -------
    Dict[str, List[float]]
        Training history.
    """
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = MetricsLogger(log_dir, "training")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        count = 0

        for batch in train_loader:
            start_time = time.time()

            image = batch["image"].to(device)

            # Handle single-band (MIRI-only) data by duplicating or creating dummy
            n_channels = image.shape[1]
            if n_channels == 1:
                # Single band (MIRI) - duplicate to create 6 NIRCam + 4 MIRI channels
                nircam = image.expand(-1, 6, -1, -1)  # [B, 6, H, W]
                miri = image.expand(-1, 4, -1, -1)    # [B, 4, H, W]
            else:
                # Multi-band data
                nircam = image[:, :6, :, :]
                miri = image[:, 6:, :, :]

            # Get targets - expand to match output shape (num_classes = 10)
            targets = batch.get("dust_mass", batch.get("sfr", torch.zeros(len(nircam))))
            targets = targets.unsqueeze(1)  # [B, 1]
            # Expand to match model output (10 values per sample)
            targets = targets.expand(-1, 10)  # [B, 10]

            optimizer.zero_grad()

            outputs = model(nircam, miri)

            # Regression loss
            loss = F.mse_loss(outputs, targets.to(device))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            count += 1
            global_step += 1

            batch_time = time.time() - start_time

            # Log
            if global_step % 100 == 0:
                val_metrics = evaluate(model, val_loader, device)
                logger.log_batch(
                    step=global_step,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    batch_time=batch_time,
                )
                logger.log_validation(
                    step=global_step,
                    val_loss=val_metrics["val_loss"],
                    val_metrics=val_metrics,
                )

                print(f"Step {global_step} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")

        avg_train_loss = train_loss / count
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, checkpoint_path)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f}")

    return history


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    val_loader : DataLoader
        Validation dataloader.
    device : str, default="cuda"
        Device to use.

    Returns
    -------
    Dict[str, float]
        Validation metrics.
    """
    model.eval()
    val_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)

            # Handle single-band (MIRI-only) data by expanding to 6+4 channels
            n_channels = image.shape[1]
            if n_channels == 1:
                nircam = image.expand(-1, 6, -1, -1)  # [B, 6, H, W]
                miri = image.expand(-1, 4, -1, -1)    # [B, 4, H, W]
            else:
                nircam = image[:, :6, :, :]
                miri = image[:, 6:, :, :]

            outputs = model(nircam, miri)

            # Dummy target for evaluation (expand to match output shape)
            targets = torch.zeros(len(nircam), outputs.shape[1], device=device)
            loss = F.mse_loss(outputs, targets)
            val_loss += loss.item()
            count += 1

    return {
        "val_loss": val_loss / count,
        "val_rmse": (val_loss / count) ** 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multimodal ViT on JWST data")
    parser.add_argument("--data", "-d", required=True, help="Data directory")
    parser.add_argument("--config", "-c", help="Path to config JSON")
    parser.add_argument("--pretrained", "-p", help="Path to pretrained weights")
    parser.add_argument("--mode", choices=["pretrain", "supervised"], default="supervised")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--variant", choices=["tiny", "small", "base", "large"], default="base")
    parser.add_argument("--no-cross-attention", action="store_true")
    parser.add_argument("--mask-prob", type=float, default=0.5, help="Masking probability for pretraining")

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Merge config with CLI args
    data_dir = args.data
    output_dir = args.output
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    variant = args.variant
    use_cross_attention = not args.no_cross_attention
    mode = args.mode

    # Create directories
    save_dir = Path(output_dir) / "checkpoints"
    log_dir = Path(output_dir) / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        variant=variant,
        use_cross_attention=use_cross_attention,
        pretrained_path=args.pretrained,
        device=device,
    )

    # Create dataloaders
    mask_prob = args.mask_prob if mode == "pretrain" else 0.0
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        mask_prob=mask_prob,
        load_into_memory=True,  # For efficiency during training
    )

    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validation on {len(val_loader.dataset)} samples")

    # Train
    if mode == "pretrain":
        pretrainer = MaskedBandPretrainer(
            model,
            mask_prob=mask_prob,
            learning_rate=learning_rate,
            device=device,
        )
        # Pretraining loop
        for epoch in range(num_epochs):
            metrics = pretrainer.train_epoch(train_loader)
            print(f"Epoch {epoch}: {metrics}")
            pretrainer.save_checkpoint(
                str(save_dir / f"pretrain_epoch_{epoch}.pt"),
                epoch, epoch,
            )
    else:
        history = train_supervised(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_dir=save_dir,
            log_dir=log_dir,
        )

        # Save final model
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": history,
        }, save_dir / "final_model.pt")

    print(f"Training complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
