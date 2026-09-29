#!/usr/bin/env python3
"""
Multi-band JWST image preprocessing.

Handles:
- Loading multiple FITS files (different bands)
- PSF alignment and homogenization
- Flux-preserving normalization (asinh stretch)
- Tile generation for multi-band stacks
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from scipy.ndimage import gaussian_filter


def load_multiband_fits(
    band_paths: Dict[str, str | Path],
    target_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load multiple band FITS files into a single multi-band array.

    Parameters
    ----------
    band_paths : Dict[str, str | Path]
        Dictionary mapping band names to FITS file paths.
    target_shape : Tuple[int, int], optional
        Target shape for alignment. If None, uses the first band's shape.

    Returns
    -------
    np.ndarray
        Multi-band array with shape (n_bands, height, width)
    Dict[str, np.ndarray]
        Dictionary of WCS headers for each band.
    """
    bands_data = {}
    bands_headers = {}

    for band_name, path in band_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FITS file not found: {path}")

        with fits.open(path) as hdul:
            science = hdul["SCI"].data
            header = hdul["SCI"].header

            # Handle potential NaN values
            science = np.nan_to_num(science, nan=0.0)

            bands_data[band_name] = science
            bands_headers[band_name] = header

    if target_shape is None:
        # Use the first band as reference
        ref_band = list(band_paths.keys())[0]
        target_shape = bands_data[ref_band].shape

    # Align all bands to target shape
    aligned_bands = {}
    for band_name, data in bands_data.items():
        if data.shape != target_shape:
            # Simple resampling - could be replaced with more sophisticated
            # WCS-based reprojection
            aligned_bands[band_name] = resample_to_shape(data, target_shape)
        else:
            aligned_bands[band_name] = data

    # Stack into multi-band array
    band_order = list(band_paths.keys())
    multi_band = np.stack([aligned_bands[band] for band in band_order], axis=0)

    return multi_band, bands_headers


def resample_to_shape(
    image: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resample an image to target shape using block averaging or interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    target_shape : Tuple[int, int]
        Target (height, width).

    Returns
    -------
    np.ndarray
        Resampled image.
    """
    h, w = image.shape
    th, tw = target_shape

    if h == th and w == tw:
        return image

    # Determine resampling factor
    if h > th and w > tw:
        # Downsample using block averaging
        h_factor = h // th
        w_factor = w // tw
        return block_reduce(image, (h_factor, w_factor), func=np.nanmean)

    # Upsample using simple interpolation (could use scipy.ndimage.zoom)
    y_ratio = th / h
    x_ratio = tw / w

    # Create output grid
    y_out = np.linspace(0, h - 1, th)
    x_out = np.linspace(0, w - 1, tw)

    # Simple bilinear interpolation
    result = np.zeros((th, tw), dtype=np.float32)
    for i in range(th):
        for j in range(tw):
            y = y_out[i]
            x = x_out[j]

            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)

            # Bilinear interpolation
            dy = y - y0
            dx = x - x0

            result[i, j] = (
                (1 - dy) * (1 - dx) * image[y0, x0] +
                (1 - dy) * dx * image[y0, x1] +
                dy * (1 - dx) * image[y1, x0] +
                dy * dx * image[y1, x1]
            )

    return result.astype(np.float32)


def asinh_normalize(
    image: np.ndarray,
    alpha: float = 0.1,
    clip_percentile: Tuple[float, float] = (1.0, 99.8)
) -> np.ndarray:
    """
    Apply asinh (soft logarithmic) normalization to preserve flux ratios.

    Based on SDSS implementation (Lupton et al.).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    alpha : float, default=0.1
        Stretch parameter. Smaller = more stretch on faint features.
    clip_percentile : Tuple[float, float], default=(1.0, 99.8)
        Percentile clipping for normalization range.

    Returns
    -------
    np.ndarray
        Normalized image in [0, 1].
    """
    # Replace infinities
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply asinh stretch
    image_stretched = np.arcsinh(image / alpha)

    # Clip and normalize to [0, 1]
    p_low, p_high = clip_percentile
    p_min = np.percentile(image_stretched, p_low)
    p_max = np.percentile(image_stretched, p_high)

    image_clipped = np.clip(image_stretched, p_min, p_max)

    # Normalize to [0, 1]
    range_val = p_max - p_min
    if range_val == 0:
        return np.zeros_like(image, dtype=np.float32)

    return ((image_clipped - p_min) / range_val).astype(np.float32)


def flux_preserving_normalize(
    multi_band: np.ndarray,
    alpha: float = 0.1,
    clip_percentile: Tuple[float, float] = (1.0, 99.8)
) -> np.ndarray:
    """
    Normalize multi-band data while preserving inter-band flux ratios.

    This is critical for SED fitting - we want to preserve the
    relative fluxes between bands.

    Parameters
    ----------
    multi_band : np.ndarray
        Multi-band array with shape (n_bands, height, width).
    alpha : float, default=0.1
        Asinh stretch parameter.
    clip_percentile : Tuple[float, float], default=(1.0, 99.8)
        Percentile clipping.

    Returns
    -------
    np.ndarray
        Normalized multi-band array.
    """
    n_bands, h, w = multi_band.shape
    normalized = np.zeros((n_bands, h, w), dtype=np.float32)

    # Normalize each band independently while preserving relative scales
    for i in range(n_bands):
        normalized[i] = asinh_normalize(
            multi_band[i], alpha=alpha, clip_percentile=clip_percentile
        )

    return normalized


def normalize_to_reference(
    multi_band: np.ndarray,
    reference_band: int = 0
) -> np.ndarray:
    """
    Normalize all bands to match the reference band's flux scale.

    This preserves the relative flux ratios between bands.

    Parameters
    ----------
    multi_band : np.ndarray
        Multi-band array (n_bands, h, w).
    reference_band : int, default=0
        Index of the reference band.

    Returns
    -------
    np.ndarray
        Normalized multi-band array.
    """
    ref_data = multi_band[reference_band]
    ref_median = np.median(np.abs(ref_data[ref_data != 0]))

    normalized = np.zeros_like(multi_band, dtype=np.float32)

    for i in range(len(multi_band)):
        if i == reference_band:
            normalized[i] = multi_band[i] / ref_median
        else:
            band_median = np.median(np.abs(multi_band[i][multi_band[i] != 0]))
            if band_median > 0:
                normalized[i] = (multi_band[i] / band_median) * ref_median

    # Clip to reasonable range
    normalized = np.clip(normalized, -10, 10)

    return normalized


def align_psf(
    image: np.ndarray,
    target_fwhm: float,
    current_fwhm: float,
    pixel_scale: float = 0.03  # arcseconds per pixel for NIRCam
) -> np.ndarray:
    """
    Align PSF of an image to a target FWHM using Gaussian convolution.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    target_fwhm : float
        Target FWHM in arcseconds.
    current_fwhm : float
        Current FWHM in arcseconds.
    pixel_scale : float, default=0.03
        Pixel scale in arcseconds/px.

    Returns
    -------
    np.ndarray
        PSF-aligned image.
    """
    # Convert FWHM to sigma (FWHM = 2*sqrt(2*ln(2))*sigma ≈ 2.355*sigma)
    current_sigma = current_fwhm / 2.355 / pixel_scale
    target_sigma = target_fwhm / 2.355 / pixel_scale

    # Only convolve if current PSF is sharper than target
    # (we can't deconvolve, only blur to match)
    if current_sigma < target_sigma:
        blur_sigma = np.sqrt(target_sigma**2 - current_sigma**2)
        return gaussian_filter(image, sigma=blur_sigma)

    return image


def preprocess_multiband(
    multi_band: np.ndarray,
    band_fwhms: Dict[str, float],
    target_fwhm: Optional[float] = None,
    normalize_method: str = "asinh",
    **kwargs
) -> np.ndarray:
    """
    Full preprocessing pipeline for multi-band data.

    Parameters
    ----------
    multi_band : np.ndarray
        Multi-band array (n_bands, h, w).
    band_fwhms : Dict[str, float]
        FWHM for each band (arcseconds).
    target_fwhm : float, optional
        Target PSF FWHM. If None, uses the sharpest band.
    normalize_method : str, default="asinh"
        Normalization method: "asinh" or "flux_preserve".
    **kwargs
        Additional arguments for normalization.

    Returns
    -------
    np.ndarray
        Preprocessed multi-band array.
    """
    n_bands, h, w = multi_band.shape
    band_names = list(band_fwhms.keys())

    if target_fwhm is None:
        target_fwhm = min(band_fwhms.values())

    # PSF alignment
    aligned_bands = np.zeros((n_bands, h, w), dtype=np.float32)
    for i, band in enumerate(band_names):
        aligned_bands[i] = align_psf(
            multi_band[i],
            target_fwhm=target_fwhm,
            current_fwhm=band_fwhms[band]
        )

    # Normalization
    if normalize_method == "asinh":
        return asinh_normalize(aligned_bands, **kwargs)
    elif normalize_method == "flux_preserve":
        return flux_preserving_normalize(aligned_bands, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")


def create_tile_dataset(
    multi_band: np.ndarray,
    output_dir: str | Path,
    tile_size: int = 256,
    stride: int = 128,
    band_names: Optional[List[str]] = None
) -> Path:
    """
    Create tile dataset from multi-band image.

    Parameters
    ----------
    multi_band : np.ndarray
        Preprocessed multi-band array (n_bands, h, w).
    output_dir : str | Path
        Output directory for tiles.
    tile_size : int, default=256
        Tile size in pixels.
    stride : int, default=128
        Stride between tiles (for overlapping tiles).
    band_names : List[str], optional
        Band names for file naming.

    Returns
    -------
    Path
        Path to tile manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bands, h, w = multi_band.shape
    if band_names is None:
        band_names = [f"band_{i}" for i in range(n_bands)]

    tiles = []
    count = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_path = output_dir / f"tile_{count:06d}.npz"

            # Extract tile for all bands
            tile_data = {
                band: multi_band[i, y:y+tile_size, x:x+tile_size]
                for i, band in enumerate(band_names)
            }

            np.savez(tile_path, **tile_data)
            count += 1
            tiles.append({
                "path": str(tile_path),
                "x": x,
                "y": y,
                "tile_size": tile_size
            })

    # Create manifest
    manifest = {
        "tile_size": tile_size,
        "stride": stride,
        "n_tiles": count,
        "n_bands": n_bands,
        "band_names": band_names,
        "tiles": tiles
    }

    manifest_path = output_dir / "manifest.json"
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
