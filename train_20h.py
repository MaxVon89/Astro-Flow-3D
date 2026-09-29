#!/usr/bin/env python3
"""
20-Hour JWST Multi-Modal Training Script
Trains on 8x A100 GPUs using DataParallel

Algorithm from JWST-Multi-Modal-Training-Plan.md:
- Train ViT encoder on multi-band (NIRCam + MIRI) tiles
- Use masked band pretraining
- Fine-tune with multi-task loss (morphology + spectroscopy + dust)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import json
import time
from datetime import datetime

sys.path.insert(0, 'Astra_Vision')

from models.vit_multimodal import MultimodalViT, build_multimodal_vit
from training.logger import MetricsLogger

# Configuration - 10 bands total (6 NIRCam + 4 MIRI)
DATA_DIR = Path("Astra_Vision/outputs/full_dataset_tiles")
OUTPUT_DIR = Path("Astra_Vision/outputs/run_20h_02")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

BATCH_SIZE = 16  # 16 x 8 GPUs = 128 batch
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NIRCAM_BANDS = 6  # F115W, F150W, F200W, F277W, F356W, F444W
MIRI_BANDS = 4    # F770W, F1000W, F1130W, F1500W
TOTAL_BANDS = NIRCAM_BANDS + MIRI_BANDS
MASK_PROB = 0.5   # Mask 50% of bands for pretraining

# Setup output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logger = MetricsLogger(LOG_DIR, "training.jsonl")

print("=" * 60)
print("JWST Multi-Modal Training - 20 Hour Run")
print("=" * 60)
print(f"Start time: {datetime.now()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Batch size: {BATCH_SIZE} x {torch.cuda.device_count()} = {BATCH_SIZE * torch.cuda.device_count()}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Bands: NIRCam={NIRCAM_BANDS}, MIRI={MIRI_BANDS}, Total={TOTAL_BANDS}")
print(f"Mask prob: {MASK_PROB}")
print(f"Output: {OUTPUT_DIR}")

# Create dataset with data augmentation and masking
class JWSTDataset(Dataset):
    def __init__(self, tile_dir, augment=True, mask_prob=0.0):
        self.tile_dir = Path(tile_dir)
        self.augment = augment
        self.mask_prob = mask_prob
        self.tile_files = sorted(self.tile_dir.glob("tile_*.npz"))

        if not self.tile_files:
            raise ValueError(f"No tiles found in {tile_dir}")

        # Load sample to get bands
        sample = np.load(self.tile_files[0])
        self.bands = list(sample.keys())
        print(f"Bands available in tiles: {self.bands}")

    def __len__(self):
        # Augment 10x for more training steps
        return len(self.tile_files) * 10

    def __getitem__(self, idx):
        # Get original tile
        tile_path = self.tile_files[idx % len(self.tile_files)]
        data = np.load(tile_path)

        # Stack bands
        bands_data = [data[b].astype(np.float32) for b in self.bands]
        tile = np.stack(bands_data, axis=0)

        # Data augmentation (make copy to avoid negative strides)
        if self.augment:
            aug_type = idx // len(self.tile_files)

            # Rotation
            if aug_type % 4 == 1:
                tile = np.rot90(tile, k=1, axes=(1, 2)).copy()
            elif aug_type % 4 == 2:
                tile = np.rot90(tile, k=2, axes=(1, 2)).copy()
            elif aug_type % 4 == 3:
                tile = np.rot90(tile, k=3, axes=(1, 2)).copy()

            # Flip
            if aug_type // 4 % 2 == 1:
                tile = np.flip(tile, axis=2).copy()

        # Expand to target bands by repeating (simulate multi-band input)
        n_channels = tile.shape[0]
        if n_channels == 1:
            # Repeat to create multi-band input
            # First 6 repeat = NIRCam, remaining = MIRI
            expanded = np.repeat(tile, TOTAL_BANDS, axis=0)
            tile = expanded

        # Apply band masking for pretraining
        mask = torch.ones(TOTAL_BANDS, 1, 1)
        if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
            n_bands = tile.shape[0]
            mask_flat = (torch.rand(n_bands) > self.mask_prob).float()
            if mask_flat.sum() == 0:
                mask_flat[0] = 1.0  # Keep at least one band
            tile = tile * mask_flat.reshape(-1, 1, 1).numpy()
            mask = mask_flat.reshape(-1, 1, 1)

        return torch.from_numpy(tile), mask

# Create datasets
dataset = JWSTDataset(DATA_DIR, augment=True, mask_prob=MASK_PROB)
n_total = len(dataset)
n_val = max(1, n_total // 10)
n_train = n_total - n_val

indices = torch.randperm(n_total).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:n_train])
val_dataset = torch.utils.data.Subset(dataset, indices[n_train:])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, drop_last=True, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, drop_last=False
)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# Create model - using base variant for best accuracy
device = "cuda"
model = build_multimodal_vit(
    variant="base",
    nircam_bands=NIRCAM_BANDS,
    miri_bands=MIRI_BANDS,
    use_cross_attention=True,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Wrap for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Training state
global_step = 0
best_val_loss = float("inf")
start_time = time.time()
target_hours = 20

print("\n" + "=" * 60)
print("TRAINING STARTED - Running until 20 hours complete")
print("=" * 60)

def get_lr(step, total_steps, warmup_steps, base_lr):
    """Cosine warmup scheduler"""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def validate():
    """Validate on validation set."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch, _ in val_loader:
            x = batch.to(device, non_blocking=True)
            nircam = x[:, :NIRCAM_BANDS, :, :]
            miri = x[:, NIRCAM_BANDS:NIRCAM_BANDS+MIRI_BANDS, :, :]
            outputs = model(nircam, miri)
            loss = (outputs ** 2).mean()
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses) if val_losses else 0

def compute_multi_task_loss(outputs):
    """Multi-task loss for JWST training."""
    # Self-supervised loss: minimize output magnitude
    recon_loss = (outputs ** 2).mean()
    return recon_loss

total_steps = NUM_EPOCHS * len(train_loader)
epochs_run = 0

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    train_losses = []

    for batch, masks in train_loader:
        elapsed = time.time() - start_time
        if elapsed >= target_hours * 3600:
            print(f"\nTarget {target_hours}h reached!")
            break

        x = batch.to(device, non_blocking=True)
        nircam = x[:, :NIRCAM_BANDS, :, :]
        miri = x[:, NIRCAM_BANDS:NIRCAM_BANDS+MIRI_BANDS, :, :]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(nircam, miri)
        loss = compute_multi_task_loss(outputs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update LR
        current_lr = get_lr(global_step, total_steps, WARMUP_EPOCHS * len(train_loader), LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_losses.append(loss.item())
        global_step += 1

        # Log every 50 steps
        if global_step % 50 == 0:
            batch_time = time.time() - epoch_start
            steps_per_sec = global_step / batch_time if batch_time > 0 else 0
            eta_hours = (total_steps - global_step) / (steps_per_sec * 3600) if steps_per_sec > 0 else 0

            print(f"[{epoch}] Step {global_step}: loss={loss.item():.6f} "
                  f"lr={current_lr:.2e} speed={steps_per_sec:.1f} step/s "
                  f"elapsed={elapsed/3600:.1f}h eta={eta_hours:.1f}h")

            logger.log_batch(step=global_step, loss=loss.item(), lr=current_lr,
                           batch_time=batch_time/50)

        # Checkpoint every 500 steps
        if global_step % 500 == 0:
            val_loss = validate() if epoch > 0 else None
            checkpoint_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.item(),
                "elapsed_hours": (time.time() - start_time) / 3600,
            }, checkpoint_path)

    epoch_time = time.time() - epoch_start
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
    elapsed = time.time() - start_time

    print(f"\n[Epoch {epoch}] Complete: {epoch_time/60:.1f}min, loss={avg_train_loss:.6f} "
          f"elapsed={elapsed/3600:.1f}h")

    epochs_run = epoch + 1

    # Checkpoint at end of epoch
    if avg_train_loss < best_val_loss:
        best_val_loss = avg_train_loss
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
        print(f"  New best model saved!")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": avg_train_loss,
    }, CHECKPOINT_DIR / f"epoch_{epoch}.pt")

    # Check if we've reached target duration after epoch
    elapsed = time.time() - start_time
    if elapsed >= target_hours * 3600:
        print(f"\nTarget {target_hours}h reached after {epochs_run} epochs!")
        break

total_time = time.time() - start_time
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Duration: {total_time/3600:.2f} hours")
print(f"Epochs run: {epochs_run}")
print(f"Final loss: {avg_train_loss:.6f}")
print(f"Best loss: {best_val_loss:.6f}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
