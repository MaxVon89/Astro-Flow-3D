from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from timm import create_model
except Exception:  # pragma: no cover
    create_model = None


@dataclass
class VisionConfig:
    model_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    image_size: int = 224
    embedding_dim: int = 192


class ViTFeatureEncoder:
    """Small local vision backbone wrapper for JWST tile feature extraction."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.model = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        if torch is None or create_model is None:
            return
        try:
            self.model = create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=self.config.embedding_dim,
            ).to(self.device)
            self.model.eval()
        except Exception:
            self.model = None

    def encode(self, batch: Any) -> Any:
        if self.model is None:
            if torch is None:
                raise RuntimeError("PyTorch is required for the vision encoder")
            return torch.zeros((len(batch), self.config.embedding_dim), device=self.device)

        with torch.no_grad():
            feature = self.model(batch.to(self.device))
        return feature


class ViTTaskHead:
    def __init__(self, embedding_dim: int = 192, num_classes: int = 3):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def __call__(self, features: Any) -> Any:
        if torch is None:
            return features
        return torch.nn.functional.softmax(features, dim=-1) if features.dim() > 1 else features
