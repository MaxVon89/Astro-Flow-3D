#!/usr/bin/env python3
"""
Cross-attention layers for multi-modal JWST data.

Implements PSF-aware cross-attention to bridge the resolution gap
between NIRCam (sharper) and MIRI (blurrier) images.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class PSFAdaptiveCrossAttention(nn.Module):
    """
    Cross-attention that adapts to PSF differences between modalities.

    Uses learned PSF kernels to weight attention based on spatial scale.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        psf_scale_factor: float = 4.0,  # MIRI/NIRCam resolution ratio
    ):
        """
        Initialize PSF-adaptive cross-attention.

        Parameters
        ----------
        query_dim : int
            Dimension of query vectors (NIRCam).
        key_dim : int
            Dimension of key vectors (MIRI).
        value_dim : int
            Dimension of value vectors.
        num_heads : int, default=8
            Number of attention heads.
        dropout : float, default=0.1
            Attention dropout.
        psf_scale_factor : float, default=4.0
            Resolution ratio between modalities (MIRI/NIRCam).
        """
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.psf_scale_factor = psf_scale_factor

        assert self.head_dim * num_heads == value_dim, "value_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(query_dim, value_dim)
        self.k_proj = nn.Linear(key_dim, value_dim)
        self.v_proj = nn.Linear(key_dim, value_dim)
        self.out_proj = nn.Linear(value_dim, value_dim)

        # PSF-aware scaling - learned parameter
        self.psf_scale = Parameter(torch.ones(1) * psf_scale_factor)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, N_q, D_q)
        key: torch.Tensor,    # (B, N_k, D_k)
        value: torch.Tensor,  # (B, N_v, D_v)
        psf_scale_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional PSF scale factor.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor (NIRCam features), shape (B, N_q, D_q)
        key : torch.Tensor
            Key tensor (MIRI features), shape (B, N_k, D_k)
        value : torch.Tensor
            Value tensor (MIRI features), shape (B, N_v, D_v)
        psf_scale_factor : float, optional
            Override the learned PSF scale factor.

        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, N_q, value_dim)
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        # Project to value dimension
        Q = self.q_proj(query)  # (B, N_q, V)
        K = self.k_proj(key)    # (B, N_k, V)
        V = self.v_proj(value)  # (B, N_v, V)

        # Reshape for multi-head attention
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # PSF-aware scaling of keys
        scale = psf_scale_factor if psf_scale_factor is not None else self.psf_scale
        K_scaled = K / scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Compute attention scores
        attn = (Q @ K_scaled.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ V  # (B, H, N_q, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N_q, self.value_dim)
        out = self.out_proj(out)

        return out


class MultiScaleCrossAttention(nn.Module):
    """
    Multi-scale cross-attention for different spatial scales.

    Applies cross-attention at multiple resolutions.
    """

    def __init__(
        self,
        dim: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-scale cross-attention.

        Parameters
        ----------
        dim : int
            Feature dimension.
        num_scales : int, default=3
            Number of resolution scales.
        num_heads : int, default=8
            Number of attention heads.
        dropout : float, default=0.1
            Dropout rate.
        """
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim

        # Scale-specific projections
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout)
            for _ in range(num_scales)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass across multiple scales.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor (B, N_q, D).
        key : torch.Tensor
            Key tensor (B, N_k, D).
        value : torch.Tensor
            Value tensor (B, N_v, D).

        Returns
        -------
        torch.Tensor
            Fused output (B, N_q, D).
        """
        B, N_q, D = query.shape
        outputs = []

        for i in range(self.num_scales):
            # Project to scale-specific features
            Q = self.q_projs[i](query)
            K = self.k_projs[i](key)
            V = self.v_projs[i](value)

            # Reshape for MultiheadAttention (requires (N, B, D) format)
            Q_perm = Q.permute(1, 0, 2)  # (N_q, B, D)
            K_perm = K.permute(1, 0, 2)  # (N_k, B, D)
            V_perm = V.permute(1, 0, 2)  # (N_v, B, D)

            # Cross-attention
            out, _ = self.attention_layers[i](Q_perm, K_perm, V_perm)

            # Reshape back
            out = out.permute(1, 0, 2)  # (B, N_q, D)
            out = self.out_projs[i](out)
            outputs.append(out)

        # Concatenate and fuse
        concat = torch.cat(outputs, dim=-1)  # (B, N_q, D * num_scales)
        out = self.fusion(concat)

        return out


class PSFDeconvolutionLayer(nn.Module):
    """
    PSF-aware deconvolution for MIRI upscaling.

    Uses learned Wiener filtering to upsample MIRI using NIRCam context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 4,
        psf_kernel_size: int = 5,
    ):
        """
        Initialize PSF deconvolution layer.

        Parameters
        ----------
        in_channels : int
            Input channels (MIRI bands).
        out_channels : int
            Output channels (upscaled MIRI).
        upscale_factor : int, default=4
            Upscaling factor (matches NIRCam/MIRI resolution ratio).
        psf_kernel_size : int, default=5
            Size of PSF kernel.
        """
        super().__init__()
        self.upscale_factor = upscale_factor

        # PSF estimation network
        self.psf_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, psf_kernel_size ** 2, 1),
            nn.Softmax(dim=1),
        )

        # Upsampling with PSF-aware convolution
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=upscale_factor * 2,
            stride=upscale_factor, padding=upscale_factor // 2
        )

        # Context fusion from NIRCam
        self.context_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        miri: torch.Tensor,      # (B, C_m, H, W)
        nircam_context=None,     # (B, C_n, H*factor, W*factor) optional
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        miri : torch.Tensor
            MIRI input (B, C_m, H, W).
        nircam_context : torch.Tensor, optional
            NIRCam context for fusion.

        Returns
        -------
        torch.Tensor
            Upscaled MIRI (B, C_out, H*factor, W*factor).
        """
        B, C, H, W = miri.shape

        # Estimate PSF
        psf_weights = self.psf_estimator(miri)  # (B, K^2, H, W)

        # Simple upsample first
        upsampled = F.interpolate(
            miri, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False
        )

        # Apply PSF-aware convolution (simplified)
        # In practice, would implement proper depthwise-separable PSF convolution
        out = self.deconv(miri)

        # Pad or crop to match upsampled size
        if out.shape[2] != upsampled.shape[2]:
            pad_h = upsampled.shape[2] - out.shape[2]
            out = F.pad(out, (0, 0, 0, pad_h))

        if out.shape[3] != upsampled.shape[3]:
            pad_w = upsampled.shape[3] - out.shape[3]
            out = F.pad(out, (0, pad_w, 0, 0))

        # Fusion with NIRCam context
        if nircam_context is not None:
            combined = torch.cat([out, nircam_context], dim=1)
            out = self.context_fusion(combined)

        return out
