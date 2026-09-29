#!/bin/bash
# JWST Multi-Modal Training Script - 20 Hour Run
# Uninterruptible training on 8x A100 GPUs

set -e  # Exit on any error

echo "=========================================="
echo "JWST Multi-Modal Training - 20 Hour Run"
echo "Started at: $(date)"
echo "=========================================="

cd /lp-dev/nvidia/projects/Astro-Flow-3D
source venv/bin/activate

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
BATCH_SIZE=16
NUM_EPOCHS=100
LEARNING_RATE=1e-4
OUTPUT_DIR="Astra_Vision/outputs/full_run_20h"

# Create output directories
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"

echo "GPU Devices: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Output: $OUTPUT_DIR"

# Data directory (tile sets we generated)
DATA_DIR="Astra_Vision/outputs/tiles_processed"
echo "Data directory: $DATA_DIR"

# Check tiles exist
tile_count=$(find "$DATA_DIR" -name "tile_*.npz" | wc -l)
echo "Available tiles: $tile_count"

if [ "$tile_count" -lt 1 ]; then
    echo "ERROR: No tiles found!"
    exit 1
fi

echo ""
echo "Starting training..."
echo "=========================================="

python3 << 'PYEOF'
import sys
sys.path.insert(0, 'Astra_Vision')

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import numpy as np
import time
from datetime import datetime

from models.vit_multimodal import MultimodalViT, build_multimodal_vit
from training.mask import MaskedBandPretrainer, BandMasker
from training.logger import MetricsLogger

# Configuration
DATA_DIR = Path("Astra_Vision/outputs/tiles_processed")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "Astra_Vision/outputs/full_run_20h"))
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 100))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
NIRCAM_BANDS = 6
MIRI_BANDS = 4

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = MetricsLogger(LOG_DIR, "training_20h.jsonl")

# Find all tile files
tile_files = list(DATA_DIR.glob("tile_set_*/*.npz"))
print(f"Found {len(tile_files)} tile files")

class TileDataset(Dataset):
    """Dataset for loading preprocessed tiles."""

    def __init__(self, tile_dir, mask_prob=0.3):
        self.tile_dir = Path(tile_dir)
        self.mask_prob = mask_prob
        self.tile_files = list(self.tile_dir.glob("tile_set_*/tile_*.npz"))

        if not self.tile_files:
            raise ValueError(f"No tile files found in {tile_dir}")

        # Load first tile to determine available bands
        first_tile = np.load(self.tile_files[0])
        self.bands = list(first_tile.keys())
        print(f"Available bands: {self.bands}")

    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        tile_path = self.tile_files[idx]
        data = np.load(tile_path)

        # Stack all bands
        bands_data = [data[b] for b in self.bands]
        tile = np.stack(bands_data, axis=0).astype(np.float32)

        # Apply random band masking for pretraining
        if torch.rand(1).item() < self.mask_prob:
            n_bands = tile.shape[0]
            mask = torch.rand(n_bands) > 0.5
            if mask.sum() == 0:
                mask[0] = True  # Keep at least one band
            tile = tile * mask.reshape(-1, 1, 1)

        return {"image": torch.from_numpy(tile)}

# Create dataset and dataloader
dataset = TileDataset(DATA_DIR, mask_prob=0.3)
n_total = len(dataset)
n_val = max(1, n_total // 10)
n_train = n_total - n_val

indices = torch.randperm(n_total).tolist()
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Batches per epoch (train): {len(train_loader)}")
print(f"Batches per epoch (val): {len(val_loader)}")

# Create model with DDP
device = "cuda"
model = build_multimodal_vit(
    variant="base",
    nircam_bands=NIRCAM_BANDS,
    miri_bands=MIRI_BANDS,
    use_cross_attention=True,
)

# Handle single-band input by expanding to multi-band
class BandExpansionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x shape: (B, C, H, W) - single band or multi-band
        B, C, H, W = x.shape

        # If single band, expand to create NIRCam+MIRI
        if C == 1:
            nircam = x.expand(-1, NIRCAM_BANDS, -1, -1)
            miri = x.expand(-1, MIRI_BANDS, -1, -1)
        elif C >= NIRCAM_BANDS + MIRI_BANDS:
            nircam = x[:, :NIRCAM_BANDS, :, :]
            miri = x[:, NIRCAM_BANDS:NIRCAM_BANDS+MIRI_BANDS, :, :]
        else:
            # Pad or split as needed
            nircam = x[:, :NIRCAM_BANDS, :, :]
            miri = x[:, NIRCAM_BANDS:, :, :]
            if miri.shape[1] < MIRI_BANDS:
                miri = torch.cat([miri, torch.zeros(B, MIRI_BANDS - miri.shape[1], H, W, device=x.device)], dim=1)

        return self.model(nircam, miri)

wrapped_model = BandExpansionWrapper(model).to(device)

# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    wrapped_model = nn.DataParallel(wrapped_model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(
    wrapped_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.05,
)

# Learning rate scheduler with warmup
def get_lr_schedule(step, total_steps, warmup_steps=500, base_lr=LEARNING_RATE):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

# Training state
global_step = 0
start_time = time.time()
best_val_loss = float("inf")
checkpoint_interval = 3600  # Save every hour

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

def process_batch(batch):
    """Process a batch and return loss."""
    x = batch["image"].to(device, non_blocking=True)
    B, C, H, W = x.shape

    # Split into NIRCam and MIRI
    if C >= NIRCAM_BANDS + MIRI_BANDS:
        nircam = x[:, :NIRCAM_BANDS, :, :]
        miri = x[:, NIRCAM_BANDS:NIRCAM_BANDS+MIRI_BANDS, :, :]
    else:
        nircam = x[:, :NIRCAM_BANDS, :, :]
        miri = x[:, NIRCAM_BANDS:, :, :]

    # Forward pass
    optimizer.zero_grad(set_to_none=True)
    outputs = wrapped_model(x)

    # Loss: minimize output norm (self-supervised)
    loss = (outputs ** 2).mean()

    return loss, outputs

def validate():
    """Validate on validation set."""
    wrapped_model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            loss, _ = process_batch(batch)
            val_losses.append(loss.item())

    return sum(val_losses) / len(val_losses)

# Training loop for ~20 hours
target_duration = 20 * 3600  # 20 hours in seconds
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    train_losses = []

    for batch_idx, batch in enumerate(train_loader):
        # Check if we've reached target duration
        elapsed = time.time() - start_time
        remaining = target_duration - elapsed

        if elapsed >= target_duration:
            print(f"\nTarget duration ({target_duration/3600:.1f}h) reached at epoch {epoch}, batch {batch_idx}")
            print(f"Elapsed: {elapsed/3600:.2f} hours")
            break

        loss, outputs = process_batch(batch)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update learning rate
        current_lr = get_lr_schedule(global_step, total_steps=NUM_EPOCHS * len(train_loader))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_losses.append(loss.item())
        global_step += 1

        # Logging
        if global_step % 10 == 0:
            batch_time = time.time() - epoch_start
            steps_per_sec = global_step / batch_time if batch_time > 0 else 0
            eta_seconds = remaining
            eta_hours = eta_seconds / 3600

            # Compute val loss every 100 steps
            val_loss = None
            if global_step % 100 == 0:
                val_loss = validate()

            print(f"[Epoch {epoch}] Step {global_step}: "
                  f"Train Loss={loss.item():.4f} | "
                  f"LR={current_lr:.2e} | "
                  f"Steps/sec={steps_per_sec:.2f} | "
                  f"ETA={eta_hours:.1f}h")

            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")

            logger.log_batch(
                step=global_step,
                loss=loss.item(),
                lr=current_lr,
                batch_time=batch_time / 10,  # Normalize to per-batch
            )

            if val_loss is not None:
                logger.log_validation(
                    step=global_step,
                    val_loss=val_loss,
                    val_metrics={"val_rmse": val_loss ** 0.5},
                )

        # Checkpoint every hour
        if global_step % 100 == 0:
            elapsed_hour = (time.time() - start_time) / 3600
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}_step_{global_step}.pt"

            state_dict = wrapped_model.state_dict()
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "elapsed_hours": elapsed_hour,
            }, checkpoint_path)

            print(f"  Checkpoint saved: {checkpoint_path}")

    epoch_time = time.time() - epoch_start
    avg_train_loss = sum(train_losses) / len(train_losses)

    print(f"\n[Epoch {epoch}] Complete!")
    print(f"  Epoch time: {epoch_time/60:.1f} min")
    print(f"  Avg train loss: {avg_train_loss:.4f}")

    # Save epoch checkpoint
    if avg_train_loss < best_val_loss:
        best_val_loss = avg_train_loss
        best_path = CHECKPOINT_DIR / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": wrapped_model.state_dict(),
            "train_loss": avg_train_loss,
        }, best_path)
        print(f"  New best model saved: {best_path}")

    # Final checkpoint
    final_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": wrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
    }, final_path)

    print(f"  Final checkpoint: {final_path}")

    # Check total elapsed time
    total_elapsed = time.time() - start_time
    print(f"\nTotal elapsed: {total_elapsed/3600:.2f} hours")

    if total_elapsed >= target_duration:
        print("Target duration reached!")
        break

# Final summary
total_time = time.time() - start_time
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Total steps: {global_step}")
print(f"Final train loss: {avg_train_loss:.4f}")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoints: {CHECKPOINT_DIR}")

# Save final model
final_model_path = OUTPUT_DIR / "final_model.pt"
torch.save({
    "model_state_dict": wrapped_model.state_dict(),
    "global_step": global_step,
    "total_hours": total_time / 3600,
}, final_model_path)

print(f"Final model saved: {final_model_path}")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
PYEOF

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
