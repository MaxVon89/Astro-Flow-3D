#!/usr/bin/env python3
"""
Complete pipeline for JWST multi-modal training.

Downloads data, preprocesses, and trains the model.
"""

import argparse
import os
import sys
from pathlib import Path


def ensure_python_path():
    """Ensure the project root is in PYTHONPATH."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def download_data(output_dir: str, bands: str) -> None:
    """Download JWST data."""
    print("=" * 60)
    print("Step 1: Downloading JWST Data")
    print("=" * 60)

    from data.download import download_ceers_data

    bands_list = [b.strip() for b in bands.split(",")]
    result = download_ceers_data(output_dir, bands_list)

    print(f"\nDownloaded {result['downloaded']} files")
    print(f"Output: {output_dir}")


def preprocess_data(
    input_dir: str,
    output_dir: str,
    tile_size: int = 256,
    stride: int = 128,
    normalization: str = "asinh",
) -> None:
    """Preprocess downloaded data."""
    print("=" * 60)
    print("Step 2: Preprocessing Data")
    print("=" * 60)

    from data.preprocess import load_multiband_fits, preprocess_multiband, create_tile_dataset

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all FITS files organized by band
    band_files = {}
    for band in ["nircam", "miri"]:
        band_files[band] = list(input_path.rglob(f"*{band}*i2d.fits"))

    print(f"Found {len(band_files['nircam'])} NIRCam files")
    print(f"Found {len(band_files['miri'])} MIRI files")

    # Use only available bands for processing
    available_bands = [band for band in ["nircam", "miri"] if len(band_files[band]) > 0]
    if not available_bands:
        print("Error: No FITS files found!")
        return

    # Process each tile set (using MIRI files as the base)
    for i, miri_file in enumerate(band_files['miri'][:5]):
        print(f"\nProcessing MIRI file {i + 1}...")

        # Load multi-band data
        band_paths = {
            "miri": miri_file,
        }

        multi_band, headers = load_multiband_fits(band_paths)

        print(f"Multi-band shape: {multi_band.shape}")

        # Preprocess (using MIRI FWHM for single-band processing)
        band_fwhms = {
            "miri": 0.24,    # arcseconds (typical MIRI FWHM)
        }

        processed = preprocess_multiband(
            multi_band,
            band_fwhms=band_fwhms,
            target_fwhm=0.24,  # Align to MIRI resolution
            normalize_method=normalization,
        )

        print(f"Processed shape: {processed.shape}")

        # Create tiles
        tile_manifest = create_tile_dataset(
            processed,
            output_path / "tiles",
            tile_size=tile_size,
            stride=stride,
            band_names=list(band_paths.keys()),
        )

        print(f"Created tiles in: {output_path / 'tiles'}")

    print(f"\nPreprocessing complete. Output: {output_path}")


def train_model(
    data_dir: str,
    output_dir: str,
    config_path: str = None,
    epochs: int = 50,
    batch_size: int = 16,
) -> None:
    """Train the multimodal ViT model."""
    print("=" * 60)
    print("Step 3: Training Model")
    print("=" * 60)

    from training.train import (
        create_dataloaders,
        create_model,
        train_supervised,
    )

    # Detect available bands from data
    data_path = Path(data_dir)
    tile_dirs = list(data_path.glob("tile_set_*"))
    nircam_bands = 0
    miri_bands = 0
    if tile_dirs:
        # Check first tile for band info
        first_tile = list(tile_dirs[0].glob("*_tiles.npz"))[0] if list(tile_dirs[0].glob("*_tiles.npz")) else None
        if first_tile:
            # Try to read band info from manifest
            manifest = tile_dirs[0] / "manifest.json"
            if manifest.exists():
                import json
                with open(manifest) as f:
                    m = json.load(f)
                    nircam_bands = m.get("nircam_bands", 0)
                    miri_bands = m.get("miri_bands", 0)

    # Create model with detected bands
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") else "cpu"
    model = create_model(
        nircam_bands=nircam_bands,
        miri_bands=miri_bands,
        variant="base",
        use_cross_attention=nircam_bands > 0 and miri_bands > 0,
        device=device,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=8,
        val_split=0.1,
        load_into_memory=True,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Train
    history = train_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epochs,
        learning_rate=1e-4,
        save_dir=Path(output_dir) / "checkpoints",
        log_dir=Path(output_dir) / "logs",
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete JWST multi-modal training pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["download", "preprocess", "train", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Directory to download data",
    )
    parser.add_argument(
        "--preprocess-dir",
        default="data/processed",
        help="Directory for preprocessed data",
    )
    parser.add_argument(
        "--train-dir",
        default="data/processed/tiles",
        help="Directory with training tiles",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--bands",
        default="nircam,miri",
        help="Comma-separated list of bands to download",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    args = parser.parse_args()

    ensure_python_path()

    if args.stage == "download" or args.stage == "all":
        download_data(args.download_dir, args.bands)

    if args.stage in ["preprocess", "all"]:
        preprocess_data(
            input_dir=args.download_dir,
            output_dir=args.preprocess_dir,
        )

    if args.stage in ["train", "all"]:
        train_model(
            data_dir=args.train_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
