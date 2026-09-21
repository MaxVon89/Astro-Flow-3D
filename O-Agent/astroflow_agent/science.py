"""Local scientific probes used by the validator, not just loss curves."""

from __future__ import annotations

import numpy as np

from src.data.preprocess import normalize


def asinh_stretch(image: np.ndarray, stretch: float) -> np.ndarray:
    """Lupton-style asinh stretch with a shared stretch parameter."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.arcsinh(image / stretch)


def shared_linear_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """Flux-preserving map: the same divisor on every band keeps ratios."""
    return np.asarray(image, dtype=np.float64) / scale


def band_ratio_error(transform_a, transform_b, band_a: np.ndarray, band_b: np.ndarray) -> float:
    valid = (band_b > 0) & np.isfinite(band_a) & np.isfinite(band_b)
    true_ratio = band_a[valid] / band_b[valid]
    mapped_a = np.asarray(transform_a(band_a), dtype=np.float64)
    mapped_b = np.asarray(transform_b(band_b), dtype=np.float64)
    mapped_ratio = mapped_a[valid] / np.clip(mapped_b[valid], 1e-12, None)
    relative = np.abs(mapped_ratio - true_ratio) / np.clip(np.abs(true_ratio), 1e-12, None)
    return float(np.median(relative))


def synthetic_two_band(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (64, 64)
    band_a = np.clip(rng.lognormal(mean=0.0, sigma=0.4, size=shape), 1e-3, None)
    temperature = rng.uniform(0.4, 1.6, size=shape)
    band_b = np.clip(band_a * temperature, 1e-3, None)
    return band_a.astype(np.float32), band_b.astype(np.float32)


def flux_ratio_report(tolerance: float = 0.01) -> dict:
    """Independent percentile-minmax vs shared linear scale vs shared asinh."""
    band_a, band_b = synthetic_two_band()
    independent = band_ratio_error(normalize, normalize, band_a, band_b)
    scale = float(np.percentile(np.concatenate([band_a.ravel(), band_b.ravel()]), 90))
    shared_linear = band_ratio_error(
        lambda img: shared_linear_scale(img, scale),
        lambda img: shared_linear_scale(img, scale),
        band_a,
        band_b,
    )
    asinh_error = band_ratio_error(
        lambda img: asinh_stretch(img, scale),
        lambda img: asinh_stretch(img, scale),
        band_a,
        band_b,
    )
    return {
        "independent_percentile_median_rel_error": independent,
        "shared_linear_median_rel_error": shared_linear,
        "shared_asinh_median_rel_error": asinh_error,
        "linear_within_tolerance": shared_linear <= tolerance,
        "percentile_destroys_ratios": independent > shared_linear,
        "tolerance": tolerance,
    }
