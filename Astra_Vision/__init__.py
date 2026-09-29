"""
Astra-Vision: Multi-modal JWST data processing and training.

This package provides tools for downloading, preprocessing,
and training on multi-band JWST observations.
"""

__version__ = "0.1.0"
__author__ = "Astra-Vision Team"

from .data.download import download_ceers_data, download_miri_deep_field
from .data.preprocess import (
    load_multiband_fits,
    preprocess_multiband,
    asinh_normalize,
)
from .models.vit_multimodal import MultimodalViT, build_multimodal_vit

__all__ = [
    "download_ceers_data",
    "download_miri_deep_field",
    "load_multiband_fits",
    "preprocess_multiband",
    "asinh_normalize",
    "MultimodalViT",
    "build_multimodal_vit",
]
