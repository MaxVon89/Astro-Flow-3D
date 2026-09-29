#!/usr/bin/env python3
"""
Checkpointing utilities for persistent training runs.
Supports atomic checkpoint writes, resume from latest checkpoint, and checkpoint validation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


class CheckpointManager:
    """Manages training checkpoints with atomic writes and resume support."""

    def __init__(
        self,
        save_dir: str | Path,
        max_checkpoints: int = 10,
        save_optimizer_state: bool = True,
        save_scheduler_state: bool = True,
        enabled: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer_state = save_optimizer_state
        self.save_scheduler_state = save_scheduler_state
        self.enabled = enabled

        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file."""
        if not self.enabled:
            return None

        checkpoint_files = list(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))

        if not checkpoint_files:
            return None

        return sorted(checkpoint_files)[-1]

    def get_checkpoint_step(self, checkpoint_path: Path) -> int:
        """Extract the step number from a checkpoint filename."""
        stem = checkpoint_path.stem
        if "step_" in stem:
            return int(stem.split("step_")[-1])
        elif "epoch_" in stem:
            return int(stem.split("epoch_")[-1])
        return 0

    def resume_from_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cuda",
    ) -> Dict[str, Any]:
        """Resume training from the latest checkpoint."""
        if not self.enabled:
            return {"step": 0, "epoch": 0, "best_metric": None, "random_state": None}

        checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path is None:
            return {"step": 0, "epoch": 0, "best_metric": None, "random_state": None}

        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if isinstance(model_state, dict):
            try:
                model.load_state_dict(model_state)
            except RuntimeError as e:
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in model_state.items():
                    if k.startswith("module."):
                        new_state[k[7:]] = v
                    else:
                        new_state["module." + k] = v
                model.load_state_dict(new_state, strict=False)

        result = {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "best_metric": checkpoint.get("best_metric", None),
            "random_state": checkpoint.get("random_state", None),
        }

        if optimizer is not None and self.save_optimizer_state:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)

        if scheduler is not None and self.save_scheduler_state:
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)

        return result

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step: int,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        best_metric: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint atomically."""
        if not self.enabled:
            return Path("")

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self._get_model_state(model),
            "best_metric": best_metric,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if optimizer is not None and self.save_optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None and self.save_scheduler_state:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metadata is not None:
            checkpoint["metadata"] = metadata

        temp_path = self.save_dir / f"checkpoint_step_{step}.pt.tmp"
        final_path = self.save_dir / f"checkpoint_step_{step}.pt"

        try:
            torch.save(checkpoint, temp_path)
            os.rename(temp_path, final_path)
            print(f"Checkpoint saved: {final_path}")
            self._cleanup_old_checkpoints()
            return final_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _get_model_state(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model state dict, handling DataParallel/DP models."""
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module.state_dict()
        return model.state_dict()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent max_checkpoints."""
        checkpoint_files = sorted(self.save_dir.glob("checkpoint_step_*.pt"))

        while len(checkpoint_files) > self.max_checkpoints:
            oldest = checkpoint_files.pop(0)
            try:
                oldest.unlink()
                print(f"Removed old checkpoint: {oldest}")
            except Exception as e:
                print(f"Warning: Could not remove {oldest}: {e}")

    def validate_checkpoint(self, checkpoint_path: Path) -> Tuple[bool, str]:
        """Validate a checkpoint file is loadable and complete."""
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            required_keys = ["step", "epoch", "model_state_dict"]
            missing = [k for k in required_keys if k not in checkpoint]

            if missing:
                return False, f"Missing keys: {missing}"

            model_state = checkpoint.get("model_state_dict", {})
            if not model_state:
                return False, "Empty model state"

            return True, "Checkpoint valid"
        except Exception as e:
            return False, f"Load error: {e}"

    def get_checkpoint_manifest(self) -> Dict[str, Any]:
        """Generate a manifest of all checkpoints."""
        checkpoints = []
        for cp_path in sorted(self.save_dir.glob("checkpoint_step_*.pt")):
            valid, msg = self.validate_checkpoint(cp_path)
            checkpoints.append({
                "path": str(cp_path),
                "step": self.get_checkpoint_step(cp_path),
                "size_mb": cp_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(cp_path.stat().st_mtime).isoformat(),
                "valid": valid,
                "validation_msg": msg,
            })

        return {
            "save_dir": str(self.save_dir),
            "max_checkpoints": self.max_checkpoints,
            "checkpoints": checkpoints,
            "total_count": len(checkpoints),
            "total_size_mb": sum(c["size_mb"] for c in checkpoints),
        }


def create_checkpoint_manager_from_config(config: Dict[str, Any]) -> CheckpointManager:
    """Create a CheckpointManager from experiment config."""
    checkpoint_config = config.get("checkpointing", {})
    return CheckpointManager(
        save_dir=checkpoint_config.get("save_dir", "/data/astroflow/checkpoints"),
        max_checkpoints=checkpoint_config.get("max_checkpoints", 10),
        save_optimizer_state=checkpoint_config.get("save_optimizer_state", True),
        save_scheduler_state=checkpoint_config.get("save_scheduler_state", True),
        enabled=checkpoint_config.get("enabled", True),
    )
