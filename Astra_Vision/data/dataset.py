#!/usr/bin/env python3
"""
PyTorch Dataset for multi-band JWST tiles.

Supports:
- Multi-band tile loading
- Masked band pretraining
- Physical parameter regression targets
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiBandTileDataset(Dataset):
    """
    Dataset for multi-band JWST tiles.

    Each sample is a multi-band tile (n_bands, h, w) with optional targets.
    """

    def __init__(
        self,
        tile_dir: str | Path,
        bands: Optional[List[str]] = None,
        mask_prob: float = 0.0,
        target_fields: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        load_into_memory: bool = False,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        tile_dir : str | Path
            Directory containing tile .npz files.
        bands : List[str], optional
            List of band names to load. If None, loads all available.
        mask_prob : float, default=0.0
            Probability of masking entire bands during training.
        target_fields : List[str], optional
            Physical parameter fields to load as targets (e.g., 'dust_mass', 'sfr').
        transform : callable, optional
            Transform function to apply to each sample.
        load_into_memory : bool, default=False
            If True, load all tiles into memory at initialization.
        """
        self.tile_dir = Path(tile_dir)
        self.mask_prob = mask_prob
        self.target_fields = target_fields or []
        self.transform = transform

        # Find all tile files
        self.tile_files = sorted(self.tile_dir.glob("tile_*.npz"))
        if not self.tile_files:
            raise ValueError(f"No tile files found in {tile_dir}")

        # Load band names from first tile
        with np.load(self.tile_files[0]) as data:
            self.available_bands = list(data.keys())
        if bands is None:
            self.bands = self.available_bands
        else:
            self.bands = [b for b in bands if b in self.available_bands]

        if not self.bands:
            raise ValueError(f"No valid bands found. Available: {self.available_bands}")

        self.n_bands = len(self.bands)

        # Optional: load all tiles into memory
        self.loaded_tiles = None
        if load_into_memory:
            print(f"Loading {len(self.tile_files)} tiles into memory...")
            self.loaded_tiles = []
            for tile_path in self.tile_files:
                with np.load(tile_path) as data:
                    tile = np.stack([data[b] for b in self.bands], axis=0)
                    self.loaded_tiles.append(tile)

        # Load targets if available
        self.targets = None
        targets_path = self.tile_dir / "targets.json"
        if targets_path.exists():
            with open(targets_path) as f:
                self.targets = json.load(f)

    def __len__(self) -> int:
        return len(self.tile_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        # Load tile
        if self.loaded_tiles is not None:
            tile = self.loaded_tiles[idx]
        else:
            with np.load(self.tile_files[idx]) as data:
                tile = np.stack([data[b] for b in self.bands], axis=0)

        # Apply masking for pretraining
        if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
            mask = torch.rand(self.n_bands) > self.mask_prob
            # At least one band must remain visible
            if not mask.any():
                mask[torch.randint(0, self.n_bands, (1,)).item()] = True
            tile = tile * mask.reshape(-1, 1, 1)

        # Convert to tensor
        tile_tensor = torch.from_numpy(tile).float()

        # Apply transform
        if self.transform:
            tile_tensor = self.transform(tile_tensor)

        # Get target if available
        sample = {"image": tile_tensor}

        if self.targets:
            tile_name = self.tile_files[idx].stem
            if tile_name in self.targets:
                target = self.targets[tile_name]
                for field in self.target_fields:
                    if field in target:
                        sample[field] = torch.tensor(target[field], dtype=torch.float32)

        return sample


class MultiBandFITSdataset(Dataset):
    """
    Dataset that loads directly from FITS files.
    Useful for large datasets that don't fit in memory.
    """

    def __init__(
        self,
        fits_dir: str | Path,
        manifest_path: Optional[str | Path] = None,
        tile_size: int = 256,
        stride: int = 256,
        bands: Optional[List[str]] = None,
        mask_prob: float = 0.0,
        transform: Optional[callable] = None,
    ):
        """
        Initialize the FITS dataset.

        Parameters
        ----------
        fits_dir : str | Path
            Directory containing FITS files (one per band).
        manifest_path : str | Path, optional
            Path to tile manifest JSON.
        tile_size : int, default=256
            Tile size.
        stride : int, default=256
            Tile stride.
        bands : List[str], optional
            Band names (must match FITS file suffixes).
        mask_prob : float, default=0.0
            Probability of masking bands.
        transform : callable, optional
            Transform function.
        """
        self.fits_dir = Path(fits_dir)
        self.tile_size = tile_size
        self.stride = stride
        self.mask_prob = mask_prob
        self.transform = transform
        self.bands = bands or []

        # Load manifest or generate it
        if manifest_path:
            self.manifest_path = Path(manifest_path)
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            self.tiles = self.manifest.get("tiles", [])
        else:
            # Generate tiles from FITS files
            self.tiles = self._generate_tiles()

        self.n_tiles = len(self.tiles)

    def _generate_tiles(self) -> List[Dict]:
        """Generate tile list from FITS files."""
        # Find all FITS files
        fits_files = {}
        for band in self.bands:
            band_files = sorted(self.fits_dir.glob(f"*{band}*"))
            if band_files:
                fits_files[band] = band_files

        if not fits_files:
            return []

        # For now, return empty - would need proper WCS alignment
        return []

    def __len__(self) -> int:
        return self.n_tiles

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from FITS files."""
        # This would load from FITS files with proper alignment
        # For now, return a placeholder
        raise NotImplementedError("FITS dataset not fully implemented")


class MaskedBandDataset:
    """
    Wrapper for masked band pretraining.

    Randomly masks bands and trains to reconstruct them.
    """

    def __init__(
        self,
        base_dataset: MultiBandTileDataset,
        mask_ratio: float = 0.2,
        mask_strategy: str = "random_band",
    ):
        """
        Initialize masked band dataset.

        Parameters
        ----------
        base_dataset : MultiBandTileDataset
            Base dataset to wrap.
        mask_ratio : float, default=0.2
            Ratio of bands to mask.
        mask_strategy : str, default="random_band"
            Masking strategy: "random_band" or "random_patch".
        """
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with band masking."""
        sample = self.base_dataset[idx]
        image = sample["image"]

        n_bands = image.shape[0]

        if self.mask_strategy == "random_band":
            # Mask random bands
            n_to_mask = max(1, int(n_bands * self.mask_ratio))
            mask_indices = torch.randperm(n_bands)[:n_to_mask]

            mask = torch.ones(n_bands, 1, 1)
            mask[mask_indices] = 0

            masked_image = image * mask
            sample["masked_image"] = masked_image
            sample["mask"] = mask
            sample["target"] = image  # Target is original image

        elif self.mask_strategy == "random_patch":
            # Mask random spatial patches
            h, w = image.shape[-2:]
            patch_size = int(min(h, w) * self.mask_ratio)
            mask = torch.ones(1, h, w)

            # Random mask position
            y = torch.randint(0, h - patch_size, (1,)).item()
            x = torch.randint(0, w - patch_size, (1,)).item()

            mask[:, y:y+patch_size, x:x+patch_size] = 0

            masked_image = image * mask
            sample["masked_image"] = masked_image
            sample["mask"] = mask
            sample["target"] = image

        return sample
