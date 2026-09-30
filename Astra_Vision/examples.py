#!/usr/bin/env python3
"""
Example scripts for running inference with the trained model.

These examples demonstrate various use cases:
1. Single tile inference
2. Batch inference
3. Loading predictions from catalog
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from models.vit_multimodal import MultimodalViT, build_multimodal_vit


def load_model(checkpoint_path: str, variant: str = "base") -> MultimodalViT:
    """
    Load a trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.pt).
    variant : str, default="base"
        Model variant.

    Returns
    -------
    MultimodalViT
        Loaded model ready for inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build model
    model = build_multimodal_vit(
        variant=variant,
        nircam_bands=6,
        miri_bands=4,
        use_cross_attention=True,
    )

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model


def load_tile(tile_path: str) -> np.ndarray:
    """
    Load a single tile from an npz file.

    Parameters
    ----------
    tile_path : str
        Path to the .npz file.

    Returns
    -------
    np.ndarray
        Loaded tile data (n_bands, height, width).
    """
    with np.load(tile_path) as data:
        available_bands = list(data.keys())
        tile = np.stack([data[b] for b in available_bands], axis=0)
    return tile


def preprocess_tile(tile: np.ndarray) -> tuple:
    """
    Preprocess tile for model inference.

    Parameters
    ----------
    tile : np.ndarray
        Input tile (n_bands, height, width).

    Returns
    -------
    tuple
        NIRCam and MIRI tensors ready for model input.
    """
    n_bands, height, width = tile.shape
    tile_tensor = torch.from_numpy(tile).float()

    if n_bands == 1:
        # Single band (MIRI) - duplicate to create 6 NIRCam + 4 MIRI channels
        nircam = tile_tensor.expand(6, -1, -1)  # [6, H, W]
        miri = tile_tensor.expand(4, -1, -1)    # [4, H, W]
    else:
        # Multi-band data
        nircam = tile_tensor[:6, :, :]
        miri = tile_tensor[6:10, :, :]

    # Add batch dimension
    nircam = nircam.unsqueeze(0)  # [1, 6, H, W]
    miri = miri.unsqueeze(0)      # [1, 4, H, W]

    return nircam, miri


def predict(model: MultimodalViT, tile: np.ndarray) -> np.ndarray:
    """
    Run inference on a single tile.

    Parameters
    ----------
    model : MultimodalViT
        Trained model.
    tile : np.ndarray
        Input tile (n_bands, height, width).

    Returns
    -------
    np.ndarray
        Predicted parameters (num_classes,).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nircam, miri = preprocess_tile(tile)
    nircam = nircam.to(device)
    miri = miri.to(device)

    with torch.no_grad():
        outputs = model(nircam, miri)

    predictions = outputs.cpu().numpy().squeeze(0)
    return predictions


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Single tile inference
    checkpoint_path = "Astra_Vision/outputs/full_run_20h/checkpoints/best_model.pt"
    tile_path = "Astra_Vision/outputs/full_dataset_tiles/tile_000000.npz"

    # Load model
    model = load_model(checkpoint_path, variant="base")

    # Load and predict on a single tile
    tile = load_tile(tile_path)
    print(f"Loaded tile shape: {tile.shape}")

    predictions = predict(model, tile)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Example 2: Load predictions from catalog
    catalog_path = "Astra_Vision/outputs/predictions.json"
    if Path(catalog_path).exists():
        with open(catalog_path) as f:
            catalog = json.load(f)
        print(f"\nCatalog contains {len(catalog)} tile predictions")
        for tile_name, data in list(catalog.items())[:3]:
            print(f"  {tile_name}: {data['predictions'][:5]}...")
