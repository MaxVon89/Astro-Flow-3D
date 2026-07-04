from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.fits_loader import load_fits
from src.data.preprocess import normalize

from src.data.reconstruction import (
    load_tiles,
    reconstruct_image,
)

from src.evaluation.image_metrics import (
    mse,
    mae,
    max_error,
)


FITS_FILE = (
    "downloads/mastDownload/JWST/"
    "jw01345003001_08201_00002_nrca2/"
    "jw01345003001_08201_00002_nrca2_i2d.fits"
)

TILE_DIR = Path(
    "data_processed/tiles"
)


print("=" * 60)
print("RECONSTRUCTION VALIDATION")
print("=" * 60)

print("\nLoading original image...")

original = load_fits(FITS_FILE)

original = normalize(original)

print("Loading tiles...")

tiles = load_tiles(TILE_DIR)

print(f"Tiles loaded: {len(tiles)}")

print("Reconstructing image...")

reconstructed, weights = reconstruct_image(
    tiles,
    original.shape,
)

# Evaluate only reconstructed pixels
mask = weights > 0

mse_value = mse(
    original[mask],
    reconstructed[mask],
)

mae_value = mae(
    original[mask],
    reconstructed[mask],
)

max_error_value = max_error(
    original[mask],
    reconstructed[mask],
)

coverage = (
    mask.sum() / mask.size
) * 100

print("\nReconstruction Summary")
print("-" * 40)

print(f"Tiles Loaded : {len(tiles)}")
print(f"Image Shape  : {original.shape}")
print(f"Coverage     : {coverage:.2f}%")
print(f"MSE          : {mse_value:.8f}")
print(f"MAE          : {mae_value:.8f}")
print(f"Max Error    : {max_error_value:.8f}")

error = np.zeros_like(original)

error[mask] = np.abs(
    original[mask] -
    reconstructed[mask]
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)

axes[0].imshow(
    original,
    cmap="gray",
    origin="lower",
)

axes[0].set_title("Original")

axes[1].imshow(
    reconstructed,
    cmap="gray",
    origin="lower",
)

axes[1].set_title("Reconstructed")

im = axes[2].imshow(
    error,
    cmap="inferno",
    origin="lower",
)

axes[2].set_title("Absolute Error")

plt.colorbar(
    im,
    ax=axes[2],
)

plt.tight_layout()

plt.savefig(
    "reconstruction_comparison.png",
    dpi=300,
)

plt.close()

plt.figure(figsize=(8,8))

plt.imshow(
    weights,
    origin="lower",
    cmap="viridis",
)

plt.colorbar()

plt.title("Weight Map")

plt.savefig(
    "reconstruction_weights.png",
    dpi=300,
)

plt.close()

print("\nSaved reconstruction_comparison.png")

print("=" * 60)