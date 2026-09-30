#!/usr/bin/env python3
"""
Inference script for the trained MultimodalViT model.

Supports:
- Single tile inference
- Batch inference
- Catalog output format
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use virtual environment if available
venv_path = PROJECT_ROOT.parent / "venv"
if venv_path.exists():
    venv_bin = venv_path / "bin"
    if (venv_bin / "python").exists():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch

from models.vit_multimodal import MultimodalViT, build_multimodal_vit


def load_model(
    checkpoint_path: str,
    variant: str = "base",
    device: str = "cuda",
) -> MultimodalViT:
    """
    Load a trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.pt).
    variant : str, default="base"
        Model variant ("tiny", "small", "base", "large").
    device : str, default="cuda"
        Device to load the model on.

    Returns
    -------
    MultimodalViT
        Loaded model ready for inference.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = {}

    # Build model
    model = build_multimodal_vit(
        variant=variant,
        nircam_bands=6,
        miri_bands=4,
        use_cross_attention=True,
        **config.get("model_args", {}),
    )

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model


def load_tile(tile_path: str, bands: Optional[List[str]] = None) -> np.ndarray:
    """
    Load a single tile from an npz file.

    Parameters
    ----------
    tile_path : str
        Path to the .npz file.
    bands : List[str], optional
        Band names to load. If None, loads all available.

    Returns
    -------
    np.ndarray
        Loaded tile data (n_bands, height, width).
    """
    with np.load(tile_path) as data:
        available_bands = list(data.keys())

        if bands is None:
            bands = available_bands

        # Stack bands along first axis
        tile = np.stack([data[b] for b in bands], axis=0)

    return tile


def preprocess_tile(
    tile: np.ndarray,
    target_nircam_bands: int = 6,
    target_miri_bands: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess tile for model inference.

    Handles single-band (MIRI-only) data by duplicating channels
    to match the expected 6 NIRCam + 4 MIRI input format.

    Parameters
    ----------
    tile : np.ndarray
        Input tile (n_bands, height, width).
    target_nircam_bands : int, default=6
        Number of NIRCam bands to create.
    target_miri_bands : int, default=4
        Number of MIRI bands to create.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        NIRCam and MIRI tensors ready for model input.
    """
    n_bands, height, width = tile.shape

    # Convert to tensor
    tile_tensor = torch.from_numpy(tile).float()

    if n_bands == 1:
        # Single band (MIRI) - duplicate to create 6 NIRCam + 4 MIRI channels
        nircam = tile_tensor.expand(target_nircam_bands, -1, -1)  # [6, H, W]
        miri = tile_tensor.expand(target_miri_bands, -1, -1)      # [4, H, W]
    else:
        # Multi-band data
        nircam = tile_tensor[:target_nircam_bands, :, :]
        miri = tile_tensor[target_nircam_bands:target_nircam_bands + target_miri_bands, :, :]

    # Add batch dimension
    nircam = nircam.unsqueeze(0)  # [1, 6, H, W]
    miri = miri.unsqueeze(0)      # [1, 4, H, W]

    return nircam, miri


def run_inference(
    model: MultimodalViT,
    tile: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run inference on a single tile.

    Parameters
    ----------
    model : MultimodalViT
        Trained model.
    tile : np.ndarray
        Input tile (n_bands, height, width).
    device : str, default="cuda"
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Predicted parameters (num_classes,).
    """
    nircam, miri = preprocess_tile(tile)

    nircam = nircam.to(device)
    miri = miri.to(device)

    with torch.no_grad():
        outputs = model(nircam, miri)

    predictions = outputs.cpu().numpy().squeeze(0)

    return predictions


def process_directory(
    model: MultimodalViT,
    tile_dir: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Process all tiles in a directory.

    Parameters
    ----------
    model : MultimodalViT
        Trained model.
    tile_dir : str
        Directory containing tile .npz files.
    output_path : str, optional
        Path to save predictions catalog (JSON or parquet).
    device : str, default="cuda"
        Device to run inference on.
    batch_size : int, default=8
        Batch size for processing.
    num_workers : int, default=4
        Number of data workers.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping tile names to predictions.
    """
    tile_dir = Path(tile_dir)
    tile_files = sorted(tile_dir.glob("tile_*.npz"))

    if not tile_files:
        raise ValueError(f"No tile files found in {tile_dir}")

    print(f"Processing {len(tile_files)} tiles...")

    predictions = {}

    for tile_path in tile_files:
        tile = load_tile(str(tile_path))
        preds = run_inference(model, tile, device)
        predictions[tile_path.stem] = preds

        if len(predictions) % 10 == 0:
            print(f"Processed {len(predictions)}/{len(tile_files)} tiles")

    # Save predictions
    if output_path:
        save_catalog(predictions, output_path)

    return predictions


def save_catalog(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    format: str = "json",
) -> None:
    """
    Save predictions to a catalog file.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping tile names to predictions.
    output_path : str
        Output file path (.json or .parquet).
    format : str, default="json"
        Output format ("json" or "parquet").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    catalog = {}
    for tile_name, preds in predictions.items():
        catalog[tile_name] = {
            "predictions": preds.tolist(),
            "num_predictions": len(preds),
        }

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"Saved catalog to {output_path}")
    elif format == "parquet":
        import pandas as pd

        df = pd.DataFrame.from_dict(
            {
                tile_name: {
                    f"pred_{i}": pred
                    for i, pred in enumerate(preds)
                }
                for tile_name, preds in predictions.items()
            },
            orient="index",
        )
        df.index.name = "tile_name"
        df.to_parquet(output_path)
        print(f"Saved catalog to {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on JWST tiles")
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input tile path or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for predictions catalog",
    )
    parser.add_argument(
        "--format",
        choices=["json", "parquet"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--variant",
        choices=["tiny", "small", "base", "large"],
        default="base",
        help="Model variant (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for directory processing (default: 8)",
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, variant=args.variant, device=args.device)

    input_path = Path(args.input)

    if input_path.is_file():
        # Single tile inference
        print(f"Loading tile: {input_path}")
        tile = load_tile(str(input_path))
        print(f"Tile shape: {tile.shape}")

        predictions = run_inference(model, tile, device=args.device)
        print(f"Predictions: {predictions}")

        if args.output:
            save_catalog({input_path.stem: predictions}, args.output, format=args.format)

    elif input_path.is_dir():
        # Batch inference
        predictions = process_directory(
            model,
            str(input_path),
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"Processed {len(predictions)} tiles")

    else:
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
