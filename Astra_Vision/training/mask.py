#!/usr/bin/env python3
"""
Masked band pretraining for multi-modal ViT.

Implements masked band reconstruction as self-supervised pretraining.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative imports work when training.mask is imported from Astra-Vision
try:
    from ..models.vit_multimodal import MultimodalViT
except ImportError:
    from models.vit_multimodal import MultimodalViT


class MaskedBandLoss(nn.Module):
    """
    Loss for masked band reconstruction.
    """

    def __init__(self, mask_weight: float = 1.0, recon_weight: float = 1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.recon_weight = recon_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked band loss.

        Parameters
        ----------
        predicted : torch.Tensor
            Predicted masked bands (B, N_bands, H, W).
        target : torch.Tensor
            Target bands (B, N_bands, H, W).
        mask : torch.Tensor
            Mask indicating which bands were masked (B, N_bands, 1, 1).

        Returns
        -------
        Dict[str, torch.Tensor]
            Loss components.
        """
        # Reconstruction loss for masked bands only
        masked_predicted = predicted * mask
        masked_target = target * mask

        recon_loss = F.mse_loss(masked_predicted, masked_target)

        # Total loss
        total_loss = self.recon_weight * recon_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
        }


class BandMasker:
    """
    Handles band masking for pretraining.
    """

    def __init__(
        self,
        n_bands: int,
        mask_prob: float = 0.5,
        mask_strategy: str = "random",
        min_keep_bands: int = 2,
    ):
        """
        Initialize band masker.

        Parameters
        ----------
        n_bands : int
            Total number of bands.
        mask_prob : float, default=0.5
            Probability of masking each band.
        mask_strategy : str, default="random"
            Masking strategy: "random", "contiguous", or "fixed".
        min_keep_bands : int, default=2
            Minimum number of bands to keep unmasked.
        """
        self.n_bands = n_bands
        self.mask_prob = mask_prob
        self.mask_strategy = mask_strategy
        self.min_keep_bands = min_keep_bands

    def generate_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate a random band mask.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        torch.Tensor
            Mask tensor (B, N_bands, 1, 1).
        """
        mask = torch.ones(batch_size, self.n_bands, 1, 1)

        if self.mask_strategy == "random":
            for b in range(batch_size):
                # Generate random mask
                mask_vec = (torch.rand(self.n_bands) > self.mask_prob).float()

                # Ensure at least min_keep_bands are visible
                if mask_vec.sum() < self.min_keep_bands:
                    visible = random.sample(range(self.n_bands), self.min_keep_bands)
                    mask_vec = torch.zeros(self.n_bands)
                    mask_vec[visible] = 1.0

                mask[b, :, 0, 0] = mask_vec

        elif self.mask_strategy == "contiguous":
            # Mask a contiguous block of bands
            for b in range(batch_size):
                mask_start = random.randint(0, self.n_bands - 2)
                mask_len = random.randint(1, self.n_bands - self.min_keep_bands)
                mask_end = min(mask_start + mask_len, self.n_bands)
                mask[b, mask_start:mask_end, 0, 0] = 0.0

        elif self.mask_strategy == "fixed":
            # Mask specific bands (e.g., always mask the last band)
            fixed_mask = torch.ones(self.n_bands)
            fixed_mask[-1] = 0.0  # Always mask last band
            mask[:, :, 0, 0] = fixed_mask

        return mask

    def apply_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply band mask to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, N_bands, H, W).
        mask : torch.Tensor, optional
            Pre-generated mask. If None, generates new mask.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Masked input and mask tensor.
        """
        if mask is None:
            mask = self.generate_mask(x.shape[0]).to(x.device)

        masked_x = x * mask
        return masked_x, mask


class MaskedBandPretrainer:
    """
    Trainer for masked band pretraining.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_prob: float = 0.5,
        mask_strategy: str = "random",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        device: str = "cuda",
    ):
        """
        Initialize pretrainer.

        Parameters
        ----------
        model : nn.Module
            Multimodal ViT model.
        mask_prob : float, default=0.5
            Band masking probability.
        mask_strategy : str, default="random"
            Masking strategy.
        learning_rate : float, default=1e-4
            Learning rate.
        weight_decay : float, default=0.05
            Weight decay.
        device : str, default="cuda"
            Device to train on.
        """
        self.model = model.to(device)
        self.masker = BandMasker(
            n_bands=model.total_bands,
            mask_prob=mask_prob,
            mask_strategy=mask_strategy,
        )
        self.loss_fn = MaskedBandLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = device

    def train_step(
        self,
        x: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Parameters
        ----------
        x : torch.Tensor
            Input bands (B, N_bands, H, W).

        Returns
        -------
        Dict[str, float]
            Loss values.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Generate mask
        mask = self.masker.generate_mask(x.shape[0]).to(self.device)

        # Apply mask
        masked_x, mask = self.masker.apply_mask(x, mask)

        # Split into NIRCam and MIRI
        n_bands = self.model.nircam_bands
        nircam = masked_x[:, :n_bands, :, :]
        miri = masked_x[:, n_bands:, :, :]

        # Forward pass
        output = self.model(nircam, miri)

        # Compute loss - treat output as reconstructed bands
        # Reshape output to match input format
        B = x.shape[0]
        reconstructed = output.reshape(B, self.model.total_bands, 16, 16)  # Example shape

        # Pad or crop to match input size
        if reconstructed.shape[2:] != x.shape[2:]:
            reconstructed = F.interpolate(
                reconstructed, size=x.shape[2:], mode='bilinear', align_corners=False
            )

        # Compute loss
        loss_dict = self.loss_fn(reconstructed, x, mask)

        # Backward pass
        loss_dict["total"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def train_epoch(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Training dataloader.

        Returns
        -------
        Dict[str, float]
            Average loss values.
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        count = 0

        for batch in dataloader:
            x = batch["image"].to(self.device)

            loss_dict = self.train_step(x)

            total_loss += loss_dict["total"]
            total_recon += loss_dict["recon"]
            count += 1

        return {
            "total_loss": total_loss / count,
            "recon_loss": total_recon / count,
        }

    def save_checkpoint(
        self,
        path: str,
        step: int,
        epoch: int,
    ):
        """Save model checkpoint."""
        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(
        self,
        path: str,
    ) -> Tuple[int, int]:
        """Load model checkpoint. Returns (step, epoch)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["step"], checkpoint["epoch"]
