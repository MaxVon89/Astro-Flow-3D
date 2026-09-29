#!/usr/bin/env python3
"""
Validation utilities for training runs.
Runs automated checks on checkpoints and generates validation reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# Run from project root for imports to work
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.experiments.checkpoint import CheckpointManager


class Validator:
    """Runs validation checks on training artifacts."""

    def __init__(self, run_id: str, artifacts_dir: str | Path):
        self.run_id = run_id
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.artifacts_dir / f"validation_report_{run_id}.json"

    def validate_checkpoint_file(self, checkpoint_path: Path) -> Tuple[bool, List[str]]:
        """Validate a checkpoint file exists and is loadable."""
        errors = []

        if not checkpoint_path.exists():
            errors.append(f"Checkpoint file not found: {checkpoint_path}")
            return False, errors

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            required_keys = ["step", "epoch", "model_state_dict"]
            for key in required_keys:
                if key not in checkpoint:
                    errors.append(f"Missing required key in checkpoint: {key}")
        except Exception as e:
            errors.append(f"Failed to load checkpoint: {e}")
            return False, errors

        return len(errors) == 0, errors

    def validate_model_state(self, checkpoint_path: Path) -> Tuple[bool, List[str]]:
        """Validate that model state in checkpoint is valid."""
        errors = []

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model_state = checkpoint.get("model_state_dict", {})

            if not model_state:
                errors.append("Empty model state dictionary")
                return False, errors

            # Check that all tensors are valid
            for name, tensor in model_state.items():
                if not isinstance(tensor, torch.Tensor):
                    errors.append(f"Invalid tensor type for {name}: {type(tensor)}")
                elif torch.isnan(tensor).any():
                    errors.append(f"NaN values found in tensor: {name}")
                elif torch.isinf(tensor).any():
                    errors.append(f"Inf values found in tensor: {name}")

        except Exception as e:
            errors.append(f"Failed to validate model state: {e}")
            return False, errors

        return len(errors) == 0, errors

    def test_forward_pass(self, checkpoint_path: Path, sample_shape: Tuple[int, ...] = (1, 4, 256, 256)) -> Tuple[bool, List[str]]:
        """Test that model can perform a forward pass."""
        errors = []

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model_state = checkpoint.get("model_state_dict", {})

            # Create model and load state
            model = self._create_test_model()
            model.load_state_dict(model_state)
            model.eval()

            # Test forward pass
            with torch.no_grad():
                sample = torch.randn(sample_shape)
                output = model(sample)

                if torch.isnan(output).any():
                    errors.append("Forward pass produced NaN output")
                elif torch.isinf(output).any():
                    errors.append("Forward pass produced Inf output")

        except Exception as e:
            errors.append(f"Forward pass test failed: {e}")
            return False, errors

        return len(errors) == 0, errors

    def validate_metrics_file(self, metrics_path: Path) -> Tuple[bool, List[str]]:
        """Validate metrics log file format."""
        errors = []

        if not metrics_path.exists():
            errors.append(f"Metrics file not found: {metrics_path}")
            return False, errors

        try:
            with open(metrics_path, "r") as f:
                lines = f.readlines()

            if not lines:
                errors.append("Metrics file is empty")
                return False, errors

            # Check first line is valid JSON
            first_line = lines[0].strip()
            if first_line:
                try:
                    data = json.loads(first_line)
                    required_keys = ["step", "timestamp", "phase"]
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"Missing key in metrics: {key}")
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON in metrics file: {e}")

        except Exception as e:
            errors.append(f"Failed to validate metrics file: {e}")
            return False, errors

        return len(errors) == 0, errors

    def compute_basic_metrics(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Compute basic metrics from checkpoint."""
        metrics = {}

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Model stats
            model_state = checkpoint.get("model_state_dict", {})
            total_params = sum(p.numel() for p in model_state.values())

            # Check parameter statistics
            param_values = torch.cat([p.flatten() for p in model_state.values() if isinstance(p, torch.Tensor)])
            metrics["total_params"] = total_params
            metrics["param_mean"] = param_values.mean().item() if param_values.numel() > 0 else 0
            metrics["param_std"] = param_values.std().item() if param_values.numel() > 0 else 0
            metrics["param_min"] = param_values.min().item() if param_values.numel() > 0 else 0
            metrics["param_max"] = param_values.max().item() if param_values.numel() > 0 else 0

            # Checkpoint info
            metrics["step"] = checkpoint.get("step", 0)
            metrics["epoch"] = checkpoint.get("epoch", 0)

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    def _create_test_model(self) -> nn.Module:
        """Create a test model for validation."""
        # This should match the training model architecture
        return nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )

    def run_all_validations(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run all validation checks."""
        report = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "status": "pending",
        }

        # Validate checkpoint file
        cp_path = checkpoint_path or self._find_latest_checkpoint()
        if cp_path:
            valid, errors = self.validate_checkpoint_file(cp_path)
            report["checks"]["checkpoint_file"] = {"valid": valid, "errors": errors}
            if not valid:
                report["status"] = "failed"

            # Validate model state
            if valid:
                valid, errors = self.validate_model_state(cp_path)
                report["checks"]["model_state"] = {"valid": valid, "errors": errors}
                if not valid:
                    report["status"] = "failed"

            # Test forward pass
            if valid:
                valid, errors = self.test_forward_pass(cp_path)
                report["checks"]["forward_pass"] = {"valid": valid, "errors": errors}
                if not valid:
                    report["status"] = "failed"

            # Compute metrics
            metrics = self.compute_basic_metrics(cp_path)
            report["metrics"] = metrics

        # Validate metrics file
        metrics_path = self.artifacts_dir / f"metrics_{self.run_id}.jsonl"
        if metrics_path.exists():
            valid, errors = self.validate_metrics_file(metrics_path)
            report["checks"]["metrics_file"] = {"valid": valid, "errors": errors}
            if not valid:
                report["status"] = "failed"

        # Write report
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in artifacts dir."""
        checkpoint_manager = CheckpointManager(self.artifacts_dir)
        return checkpoint_manager.get_latest_checkpoint()


def validate_run(run_id: str, artifacts_dir: str | Path, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate a run."""
    validator = Validator(run_id, artifacts_dir)
    cp_path = Path(checkpoint_path) if checkpoint_path else None
    return validator.run_all_validations(cp_path)


def main():
    parser = argparse.ArgumentParser(description="Validate training artifacts")
    parser.add_argument("--run-id", "-r", required=True, help="Run ID")
    parser.add_argument("--artifacts-dir", "-a", default="/data/astroflow/checkpoints", help="Artifacts directory")
    parser.add_argument("--checkpoint", "-c", help="Specific checkpoint to validate")
    parser.add_argument("--output", "-o", help="Output file for report")
    args = parser.parse_args()

    report = validate_run(args.run_id, args.artifacts_dir, args.checkpoint)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    # Exit with error code if validation failed
    if report.get("status") != "pass":
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()
