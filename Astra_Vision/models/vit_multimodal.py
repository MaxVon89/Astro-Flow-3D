#!/usr/bin/env python3
"""
Multimodal Vision Transformer for JWST NIRCam+MIRI data.

Implements a cross-attention based ViT that jointly processes
NIRCam (sharper) and MIRI (blurrier) images.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import PSFAdaptiveCrossAttention, MultiScaleCrossAttention


class PatchEmbedding(nn.Module):
    """
    Patch embedding for image tokens.

    Projects image patches to embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 10,  # 6 NIRCam + 4 MIRI
        embed_dim: int = 768,
    ):
        """
        Initialize patch embedding.

        Parameters
        ----------
        img_size : int, default=256
            Image size (square).
        patch_size : int, default=16
            Patch size (square).
        in_channels : int, default=10
            Number of input channels (bands).
        embed_dim : int, default=768
            Embedding dimension.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Patch embeddings with cls token (B, N_patches + 1, embed_dim).
        """
        B, C, H, W = x.shape

        # Project to embedding
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)

        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        return x


class Attention(nn.Module):
    """
    Multi-head self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP block for Transformer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block with self-attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Transformer block with cross-attention between NIRCam and MIRI.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        psf_scale_factor: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)

        # Self-attention for NIRCam
        self.nircam_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        # Cross-attention from MIRI to NIRCam
        self.cross_attn = PSFAdaptiveCrossAttention(
            query_dim=dim,
            key_dim=dim,
            value_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            psf_scale_factor=psf_scale_factor,
        )

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, drop=drop)

    def forward(
        self,
        nircam_tokens: torch.Tensor,  # (B, N_n, D)
        miri_tokens: torch.Tensor,    # (B, N_m, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention.

        Parameters
        ----------
        nircam_tokens : torch.Tensor
            NIRCam token embeddings (B, N_n, D).
        miri_tokens : torch.Tensor
            MIRI token embeddings (B, N_m, D).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated NIRCam and MIRI tokens.
        """
        # Self-attention on NIRCam
        nircam_tokens = nircam_tokens + self.nircam_attn(self.norm1(nircam_tokens))

        # Cross-attention: MIRI -> NIRCam
        # NIRCam queries, MIRI keys and values
        nircam_cross = self.cross_attn(
            self.norm_cross(nircam_tokens),
            miri_tokens,
            miri_tokens,
        )
        nircam_tokens = nircam_tokens + nircam_cross

        # MLP on NIRCam
        nircam_tokens = nircam_tokens + self.mlp(self.norm3(nircam_tokens))

        # MIRI gets updated via feedback from NIRCam
        # (simplified - could add feedback attention)
        return nircam_tokens, miri_tokens


class MultimodalViT(nn.Module):
    """
    Multimodal ViT for NIRCam+MIRI joint processing.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        nircam_bands: int = 6,
        miri_bands: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,  # Physical parameters to predict
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        use_cross_attention: bool = True,
        psf_scale_factor: float = 4.0,  # MIRI/NIRCam resolution ratio
    ):
        """
        Initialize Multimodal ViT.

        Parameters
        ----------
        img_size : int, default=256
            Input image size.
        patch_size : int, default=16
            Patch size.
        nircam_bands : int, default=6
            Number of NIRCam bands.
        miri_bands : int, default=4
            Number of MIRI bands.
        embed_dim : int, default=768
            Embedding dimension.
        depth : int, default=12
            Number of Transformer blocks.
        num_heads : int, default=12
            Number of attention heads.
        mlp_ratio : float, default=4.0
            MLP hidden size ratio.
        num_classes : int, default=10
            Number of output classes/parameters.
        drop_rate : float, default=0.1
            Dropout rate.
        attn_drop_rate : float, default=0.1
            Attention dropout rate.
        use_cross_attention : bool, default=True
            Whether to use cross-attention between modalities.
        psf_scale_factor : float, default=4.0
            PSF scale factor for cross-attention.
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.nircam_bands = nircam_bands
        self.miri_bands = miri_bands
        self.total_bands = nircam_bands + miri_bands
        self.use_cross_attention = use_cross_attention

        # Patch embedding for each modality (use at least 1 channel for compatibility)
        self.nircam_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=max(1, nircam_bands),
            embed_dim=embed_dim,
        )
        self.miri_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=max(1, miri_bands),
            embed_dim=embed_dim,
        )

        # Channel adapters to unify dimensions
        self.nircam_adapter = nn.Linear(embed_dim, embed_dim)
        self.miri_adapter = nn.Linear(embed_dim, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if use_cross_attention and i % 2 == 1:  # Cross-attention every other layer
                self.blocks.append(CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    psf_scale_factor=psf_scale_factor,
                ))
            else:
                self.blocks.append(TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                ))

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head (uses CLS token only)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # 2 CLS tokens (one per modality)
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        nircam: torch.Tensor,  # (B, N_c, H, W)
        miri: torch.Tensor,    # (B, N_m, H, W)
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        nircam : torch.Tensor
            NIRCam input (B, N_c, H, W).
        miri : torch.Tensor
            MIRI input (B, N_m, H, W).

        Returns
        -------
        torch.Tensor
            Predicted parameters (B, num_classes).
        """
        # Ensure inputs are float32 (DataLoader might pass float64)
        nircam = nircam.float()
        miri = miri.float()

        B = nircam.shape[0]

        # Embed patches
        nircam_tokens = self.nircam_embed(nircam)  # (B, N + 1, D)
        miri_tokens = self.miri_embed(miri)        # (B, N + 1, D)

        # Adapter
        nircam_tokens = self.nircam_adapter(nircam_tokens)
        miri_tokens = self.miri_adapter(miri_tokens)

        # Process through blocks
        nircam_cls = nircam_tokens[:, :1, :]  # CLS token
        miri_cls = miri_tokens[:, :1, :]      # CLS token

        for block in self.blocks:
            if self.use_cross_attention and isinstance(block, CrossAttentionBlock):
                nircam_cls, miri_cls = block(nircam_cls, miri_cls)
            else:
                # Process CLS tokens together
                combined = torch.cat([nircam_cls, miri_cls], dim=1)
                combined = block(combined)
                nircam_cls = combined[:, :1, :]
                miri_cls = combined[:, 1:2, :]

        # Use CLS tokens for prediction
        combined_cls = torch.cat([nircam_cls, miri_cls], dim=-1)  # (B, 1, 2*embed_dim)
        combined_cls = combined_cls.squeeze(1)  # (B, 2*embed_dim)
        x = self.head(combined_cls)

        return x


def build_multimodal_vit(
    variant: str = "base",
    **kwargs,
) -> MultimodalViT:
    """
    Build a Multimodal ViT with predefined configuration.

    Parameters
    ----------
    variant : str, default="base"
        Model variant: "tiny", "small", "base", "large".
    **kwargs
        Override configuration parameters.

    Returns
    -------
    MultimodalViT
        Configured model.
    """
    configs = {
        "tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
        },
        "small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
        "base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
        },
        "large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
        },
    }

    config = configs[variant].copy()
    config.update(kwargs)

    return MultimodalViT(**config)
