#!/usr/bin/env python3
"""
Example usage of Astra-Vision for JWST multi-modal training.

This script demonstrates:
1. Downloading JWST data from MAST
2. Preprocessing multi-band data
3. Training a multimodal ViT model
"""

import os
import sys
from pathlib import Path

# Add Astra-Vision to path
sys.path.insert(0, str(Path(__file__).parent))


def example_download_data():
    """Example: Download JWST CEERS data."""
    print("=" * 60)
    print("Example: Download JWST Data")
    print("=" * 60)

    from data.download import download_ceers_data

    output_dir = "downloads/ceers"
    result = download_ceers_data(output_dir, bands=["nircam", "miri"])

    print(f"\nDownloaded {result['downloaded']} files")
    print(f"Output: {output_dir}")
    return result


def example_preprocess_data():
    """Example: Preprocess downloaded data."""
    print("=" * 60)
    print("Example: Preprocess Data")
    print("=" * 60)

    from data.preprocess import (
        load_multiband_fits,
        preprocess_multiband,
        create_tile_dataset,
    )

    # This would use real downloaded data
    print("For a real example, first run example_download_data()")

    # Example with dummy data
    import numpy as np

    # Create dummy multi-band data
    nircam_data = np.random.randn(6, 256, 256).astype(np.float32)
    miri_data = np.random.randn(4, 256, 256).astype(np.float32)

    # Stack into multi-band array
    multi_band = np.concatenate([nircam_data, miri_data], axis=0)
    print(f"Multi-band shape: {multi_band.shape}")

    # Preprocess
    band_fwhms = {
        "nircam": 0.06,
        "miri": 0.24,
    }

    processed = preprocess_multiband(
        multi_band,
        band_fwhms=band_fwhms,
        target_fwhm=0.24,
        normalize_method="asinh",
    )

    print(f"Processed shape: {processed.shape}")

    # Create tiles
    output_dir = Path("data/tiles")
    output_dir.mkdir(exist_ok=True)

    manifest = create_tile_dataset(
        processed,
        output_dir / "example",
        tile_size=64,
        stride=32,
        band_names=["nircam", "miri"],
    )

    print(f"Created {len(list(output_dir.glob('tile_set_*')))} tile sets")
    return processed


def example_train_model():
    """Example: Train multimodal ViT model."""
    print("=" * 60)
    print("Example: Train Model")
    print("=" * 60)

    import torch
    from models.vit_multimodal import MultimodalViT
    from training.mask import BandMasker, MaskedBandLoss, MaskedBandPretrainer

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalViT(
        nircam_bands=6,
        miri_bands=4,
        img_size=256,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=6,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy data loader
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 10
    nircam_data = torch.randn(n_samples, 6, 256, 256)
    miri_data = torch.randn(n_samples, 4, 256, 256)

    dataset = TensorDataset(nircam_data, miri_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Pretrain with masked bands
    pretrainer = MaskedBandPretrainer(
        model,
        mask_prob=0.5,
        learning_rate=1e-4,
        device=device,
    )

    # Train for a few steps
    print("\nStarting pretraining...")
    for epoch in range(2):
        metrics = pretrainer.train_epoch(dataloader)
        print(f"Epoch {epoch}: {metrics}")

    print("\nPretraining complete!")
    return model


def example_full_pipeline():
    """Run a minimal example of the full pipeline."""
    print("=" * 60)
    print("Example: Full Pipeline")
    print("=" * 60)

    import torch
    from models.vit_multimodal import MultimodalViT

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalViT(
        nircam_bands=6,
        miri_bands=4,
        img_size=256,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=6,
    ).to(device)

    # Create sample data
    nircam = torch.randn(2, 6, 256, 256).to(device)
    miri = torch.randn(2, 4, 256, 256).to(device)

    # Forward pass
    output = model(nircam, miri)

    print(f"Input NIRCam: {nircam.shape}")
    print(f"Input MIRI: {miri.shape}")
    print(f"Output: {output.shape}")
    print("\nPipeline example complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astra-Vision examples")
    parser.add_argument("--example", choices=["download", "preprocess", "train", "pipeline"], default="pipeline")
    args = parser.parse_args()

    if args.example == "download":
        example_download_data()
    elif args.example == "preprocess":
        example_preprocess_data()
    elif args.example == "train":
        example_train_model()
    elif args.example == "pipeline":
        example_full_pipeline()
