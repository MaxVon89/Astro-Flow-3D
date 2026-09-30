#!/usr/bin/env python3
"""
Evaluation script for the trained Multimodal ViT model.

Computes metrics and generates qualitative analysis report.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MultiBandTileDataset
from models.vit_multimodal import MultimodalViT, build_multimodal_vit


def create_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.1):
    """Create training and validation dataloaders from dataset."""
    dataset = MultiBandTileDataset(tile_dir=data_dir, load_into_memory=True)

    # Split into train/val
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    import torch
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Use subsets directly
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No workers needed for in-memory dataset
    )

    return val_loader, dataset


def load_model(checkpoint_path: str, device: str = "cuda") -> MultimodalViT:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get model config from checkpoint if available, else use defaults
    model = MultimodalViT(
        img_size=256,
        patch_size=16,
        nircam_bands=6,
        miri_bands=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=10,
    )

    # Load state dict (handle both DataParallel and "model." prefix)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Remove "model." prefix if present
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Remove "module." prefix if present (DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load state dict with strict=False to handle potential mismatches
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model: nn.Module, val_loader, device: str = "cuda"):
    """
    Evaluate model on validation set and collect predictions/targets.

    Returns dict with predictions, targets, and metrics.
    """
    all_predictions = []
    all_targets = []
    all_images = []

    val_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)

            # Handle single-band (MIRI-only) data by expanding to 6+4 channels
            n_channels = image.shape[1]
            if n_channels == 1:
                nircam = image.expand(-1, 6, -1, -1)
                miri = image.expand(-1, 4, -1, -1)
            else:
                nircam = image[:, :6, :, :]
                miri = image[:, 6:, :, :]

            outputs = model(nircam, miri)

            # Targets (zero-initialized since original dataset has no targets)
            targets = torch.zeros(len(nircam), outputs.shape[1], device=device)

            loss = nn.functional.mse_loss(outputs, targets)
            val_loss += loss.item()
            count += 1

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_images.append(image.cpu().numpy())

    # Flatten results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    images = np.concatenate(all_images, axis=0)

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    # Per-parameter metrics
    param_mse = np.mean((predictions - targets) ** 2, axis=0)
    param_mae = np.mean(np.abs(predictions - targets), axis=0)
    param_rmse = np.sqrt(param_mse)

    return {
        "predictions": predictions,
        "targets": targets,
        "images": images,
        "val_loss": val_loss / count,
        "val_rmse": np.sqrt(val_loss / count),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "param_mse": param_mse,
        "param_mae": param_mae,
        "param_rmse": param_rmse,
        "n_samples": len(predictions),
    }


def generate_report(metrics: dict, checkpoint_path: str, log_path: str, output_path: str):
    """Generate qualitative analysis report."""
    report_lines = []

    # Header
    report_lines.append("# Model Evaluation Report\n")
    report_lines.append("## Summary\n")
    report_lines.append(f"**Model:** Multimodal ViT (base variant)\n")
    report_lines.append(f"**Checkpoint:** `{checkpoint_path}`\n")

    # Metrics
    report_lines.append("\n## Quantitative Metrics\n")
    report_lines.append("### Overall Metrics")
    report_lines.append("\n| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| MSE | {metrics['mse']:.8f} |")
    report_lines.append(f"| MAE | {metrics['mae']:.8f} |")
    report_lines.append(f"| RMSE | {metrics['rmse']:.8f} |")
    report_lines.append(f"| Validation Loss | {metrics['val_loss']:.8f} |")
    report_lines.append(f"| Number of Samples | {metrics['n_samples']} |")

    # Per-parameter metrics
    report_lines.append("\n### Per-Parameter Metrics (10 output dimensions)")
    report_lines.append("\n| Parameter | MSE | MAE | RMSE |")
    report_lines.append("|-----------|-----|-----|------|")
    for i, (mse, mae, rmse) in enumerate(zip(metrics['param_mse'], metrics['param_mae'], metrics['param_rmse'])):
        report_lines.append(f"| dim_{i} | {mse:.8f} | {mae:.8f} | {rmse:.8f} |")

    # Training/Validation Loss Curves
    report_lines.append("\n## Training Curves\n")

    # Parse JSONL log file
    train_loss = []
    val_loss = []
    steps = []
    val_steps = []

    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            step = entry.get('step', 0)
            phase = entry.get('phase', '')

            if phase == 'train' and 'loss' in entry:
                train_loss.append(entry['loss'])
                steps.append(step)
            elif phase == 'validation' and 'val_loss' in entry:
                val_loss.append(entry['val_loss'])
                val_steps.append(step)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label='Training Loss', alpha=0.7)
    plt.scatter(val_steps, val_loss, label='Validation Loss', color='red', zorder=5)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    loss_plot_path = output_path.replace('.md', '_loss_curve.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    report_lines.append(f"\n![Training/Validation Loss Curve](./{os.path.basename(loss_plot_path)})\n")

    # Prediction Distributions
    report_lines.append("\n## Prediction Distributions\n")

    # Plot prediction histograms for each dimension
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    pred_flat = metrics['predictions'].flatten()
    target_flat = metrics['targets'].flatten()

    for i in range(10):
        pred_i = metrics['predictions'][:, i]
        axes[i].hist(pred_i, bins=50, alpha=0.7, label=f'dim_{i} pred', color='blue')
        axes[i].hist(metrics['targets'][:, i], bins=50, alpha=0.7, label=f'dim_{i} target', color='orange')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Dimension {i}')
        axes[i].legend(fontsize=6)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_plot_path = output_path.replace('.md', '_distributions.png')
    plt.savefig(hist_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    report_lines.append(f"\n![Prediction Distributions](./{os.path.basename(hist_plot_path)})\n")

    # Sample Predictions
    report_lines.append("\n## Sample Predictions\n")

    # Select some samples to visualize
    n_samples_show = min(5, len(metrics['predictions']))
    sample_plot_path = output_path.replace('.md', '_samples.png')

    fig, axes = plt.subplots(n_samples_show, 2, figsize=(10, 3 * n_samples_show))
    if n_samples_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples_show):
        # Show first input band (image)
        img = metrics['images'][i, 0]
        axes[i, 0].imshow(img, cmap='viridis')
        axes[i, 0].set_title(f'Sample {i} - Input (Band 0)')
        axes[i, 0].axis('off')

        # Show prediction vs target bar chart
        pred = metrics['predictions'][i]
        target = metrics['targets'][i]
        x = np.arange(10)
        width = 0.35
        axes[i, 1].bar(x - width/2, pred, width, label='Prediction', color='blue', alpha=0.7)
        axes[i, 1].bar(x + width/2, target, width, label='Target', color='orange', alpha=0.7)
        axes[i, 1].set_xlabel('Output Dimension')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].set_title(f'Sample {i} - Predictions vs Targets')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(sample_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    report_lines.append(f"\n![Sample Predictions](./{os.path.basename(sample_plot_path)})\n")

    # Additional Statistics
    report_lines.append("\n## Additional Statistics\n")
    report_lines.append(f"\n**Prediction Statistics:**")
    report_lines.append(f"- Mean: {np.mean(pred_flat):.6f}")
    report_lines.append(f"- Std: {np.std(pred_flat):.6f}")
    report_lines.append(f"- Min: {np.min(pred_flat):.6f}")
    report_lines.append(f"- Max: {np.max(pred_flat):.6f}")

    report_lines.append(f"\n**Target Statistics:**")
    report_lines.append(f"- Mean: {np.mean(target_flat):.6f}")
    report_lines.append(f"- Std: {np.std(target_flat):.6f}")
    report_lines.append(f"- Min: {np.min(target_flat):.6f}")
    report_lines.append(f"- Max: {np.max(target_flat):.6f}")

    # Correlation analysis
    report_lines.append("\n### Correlation Between Output Dimensions")
    corr_matrix = np.corrcoef(metrics['predictions'].T)
    # Handle NaN correlations (from zero-variance cases)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    report_lines.append("\n|",)
    for j in range(10):
        report_lines.append(f" dim_{j} |")
    report_lines.append("")
    report_lines.append("|" + "|".join(["------"] * 10) + "|")

    for i in range(10):
        row = f" dim_{i} |"
        for j in range(10):
            row += f" {corr_matrix[i, j]:.3f} |"
        report_lines.append(row)

    report_lines.append("\n---\n")
    report_lines.append("\n*Report generated on evaluation*\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return output_path


def main():
    """Main evaluation function."""
    # Paths
    checkpoint_path = "/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/outputs/full_run_20h/checkpoints/best_model.pt"
    data_dir = "/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/outputs/full_dataset_tiles"
    log_path = "/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/outputs/full_run_20h/logs/training_20h.jsonl"
    output_path = "/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/outputs/evaluation_report.md"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloader
    print("Creating validation dataloader...")
    val_loader, full_dataset = create_dataloaders(data_dir, batch_size=16, val_split=0.1)
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully")

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, val_loader, device)
    print(f"Evaluation complete!")
    print(f"  MSE: {metrics['mse']:.8f}")
    print(f"  MAE: {metrics['mae']:.8f}")
    print(f"  RMSE: {metrics['rmse']:.8f}")
    print(f"  Val Loss: {metrics['val_loss']:.8f}")

    # Generate report
    print(f"Generating report at {output_path}...")
    generate_report(metrics, checkpoint_path, log_path, output_path)
    print("Report generated successfully!")

    # Also save metrics as JSON
    metrics_output = output_path.replace('.md', '_metrics.json')
    with open(metrics_output, 'w') as f:
        json.dump({
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'val_loss': float(metrics['val_loss']),
            'val_rmse': float(metrics['val_rmse']),
            'n_samples': int(metrics['n_samples']),
            'param_mse': [float(x) for x in metrics['param_mse']],
            'param_mae': [float(x) for x in metrics['param_mae']],
            'param_rmse': [float(x) for x in metrics['param_rmse']],
        }, f, indent=2)
    print(f"Metrics saved to {metrics_output}")


if __name__ == "__main__":
    main()
