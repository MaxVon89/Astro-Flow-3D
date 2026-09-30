#!/usr/bin/env python3
"""
Batch inference script for processing many tiles efficiently.

Supports:
- Memory-efficient batch processing
- Progress tracking
- Configurable batch size and workers
- Parallel tile loading
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
import torch.nn.functional as F
from tqdm import tqdm

from models.vit_multimodal import MultimodalViT, build_multimodal_vit


def load_tile_data(tile_path: str) -> Tuple[str, np.ndarray]:
    """
    Load tile data from file.

    Parameters
    ----------
    tile_path : str
        Path to the .npz file.

    Returns
    -------
    Tuple[str, np.ndarray]
        Tile name and data.
    """
    with np.load(tile_path) as data:
        available_bands = list(data.keys())
        # Load all bands
        tile = np.stack([data[b] for b in available_bands], axis=0)
    return Path(tile_path).stem, tile


def preprocess_batch(
    tiles: List[np.ndarray],
    target_nircam_bands: int = 6,
    target_miri_bands: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a batch of tiles for model inference.

    Parameters
    ----------
    tiles : List[np.ndarray]
        List of tile arrays (each with shape n_bands, H, W).
    target_nircam_bands : int, default=6
        Number of NIRCam bands to create.
    target_miri_bands : int, default=4
        Number of MIRI bands to create.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        NIRCam and MIRI tensors with batch dimension.
    """
    batch_size = len(tiles)

    nircam_list = []
    miri_list = []

    for tile in tiles:
        n_bands, height, width = tile.shape
        tile_tensor = torch.from_numpy(tile).float()

        if n_bands == 1:
            # Single band (MIRI) - duplicate to create 6 NIRCam + 4 MIRI channels
            nircam = tile_tensor.expand(target_nircam_bands, -1, -1)
            miri = tile_tensor.expand(target_miri_bands, -1, -1)
        else:
            # Multi-band data
            nircam = tile_tensor[:target_nircam_bands, :, :]
            miri = tile_tensor[target_nircam_bands:target_nircam_bands + target_miri_bands, :, :]

        nircam_list.append(nircam)
        miri_list.append(miri)

    nircam_batch = torch.stack(nircam_list)  # [B, 6, H, W]
    miri_batch = torch.stack(miri_list)       # [B, 4, H, W]

    return nircam_batch, miri_batch


def run_batch_inference(
    model: MultimodalViT,
    tiles: List[np.ndarray],
    device: str = "cuda",
    batch_size: int = 8,
) -> List[np.ndarray]:
    """
    Run inference on a batch of tiles.

    Parameters
    ----------
    model : MultimodalViT
        Trained model.
    tiles : List[np.ndarray]
        List of tile arrays.
    device : str, default="cuda"
        Device to run inference on.
    batch_size : int, default=8
        Batch size for inference.

    Returns
    -------
    List[np.ndarray]
        List of prediction arrays.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            nircam, miri = preprocess_batch(batch_tiles)

            nircam = nircam.to(device)
            miri = miri.to(device)

            outputs = model(nircam, miri)
            predictions = outputs.cpu().numpy()

            for pred in predictions:
                all_predictions.append(pred)

    return all_predictions


def get_tile_files(tile_dir: str, pattern: str = "tile_*.npz") -> List[Path]:
    """
    Get list of tile files in a directory.

    Parameters
    ----------
    tile_dir : str
        Directory containing tiles.
    pattern : str, default="tile_*.npz"
        File pattern to match.

    Returns
    -------
    List[Path]
        List of tile file paths.
    """
    tile_dir = Path(tile_dir)
    return sorted(tile_dir.glob(pattern))


def process_directory_batch(
    model: MultimodalViT,
    tile_dir: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
    num_workers: int = 4,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Process all tiles in a directory with batching.

    Parameters
    ----------
    model : MultimodalViT
        Trained model.
    tile_dir : str
        Directory containing tile .npz files.
    output_path : str, optional
        Path to save predictions catalog.
    device : str, default="cuda"
        Device to run inference on.
    batch_size : int, default=8
        Batch size for inference.
    num_workers : int, default=4
        Number of workers for parallel loading.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping tile names to predictions.
    """
    tile_dir = Path(tile_dir)

    # Get tile files
    tile_files = get_tile_files(str(tile_dir))
    if not tile_files:
        raise ValueError(f"No tile files found in {tile_dir}")

    if verbose:
        print(f"Found {len(tile_files)} tile files")

    # Load tiles in parallel
    if verbose:
        print(f"Loading tiles with {num_workers} workers...")

    all_tiles = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_tile_data, str(f)): f for f in tile_files}

        for future in tqdm(futures, total=len(futures), desc="Loading", disable=not verbose):
            tile_name, tile = future.result()
            all_tiles[tile_name] = tile

    # Process in batches
    if verbose:
        print(f"Running inference with batch size {batch_size}...")

    predictions = {}
    tile_list = list(all_tiles.items())
    total_batches = (len(tile_list) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Inference", disable=not verbose):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tile_list))

        batch_items = tile_list[start_idx:end_idx]
        batch_names = [name for name, _ in batch_items]
        batch_tiles = [tile for _, tile in batch_items]

        batch_preds = run_batch_inference(model, batch_tiles, device, batch_size)

        for name, pred in zip(batch_names, batch_preds):
            predictions[name] = pred

    # Save predictions
    if output_path:
        save_catalog(predictions, output_path, verbose=verbose)

    return predictions


def save_catalog(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    format: str = "json",
    verbose: bool = True,
) -> None:
    """
    Save predictions to a catalog file.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping tile names to predictions.
    output_path : str
        Output file path.
    format : str, default="json"
        Output format ("json" or "parquet").
    verbose : bool, default=True
        Print status messages.
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
        if verbose:
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
        if verbose:
            print(f"Saved catalog to {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}")


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
        Model variant.
    device : str, default="cuda"
        Device to load the model on.

    Returns
    -------
    MultimodalViT
        Loaded model ready for inference.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config from checkpoint
    config = {}
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_args" in checkpoint:
        config = checkpoint.get("model_args", {})

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


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference on JWST tiles"
    )
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
        help="Input directory containing tiles",
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
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for loading (default: 4)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, variant=args.variant, device=args.device)

    # Process directory
    predictions = process_directory_batch(
        model,
        args.input,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=not args.quiet,
    )

    print(f"\nProcessed {len(predictions)} tiles")
    if predictions:
        example_pred = next(iter(predictions.values()))
        print(f"Predictions shape: {example_pred.shape}")


if __name__ == "__main__":
    main()
