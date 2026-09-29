#!/usr/bin/env python3
"""
Metrics logger for training.

Writes metrics in JSON-lines format for easy analysis.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsLogger:
    """
    Logger for training metrics in JSON-lines format.
    """

    def __init__(self, log_dir: str | Path, filename: str = "metrics.jsonl"):
        """
        Initialize logger.

        Parameters
        ----------
        log_dir : str | Path
            Directory to write logs.
        filename : str, default="metrics.jsonl"
            Log filename.
        """
        self.log_dir = Path(log_dir)
        self.metrics_path = self.log_dir / filename
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Open file for appending
        self._file = open(self.metrics_path, "a")

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a metrics dict.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics to log.
        """
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def log_batch(
        self,
        step: int,
        loss: float,
        lr: float,
        batch_time: float,
        gpu_mem_mb: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Log a training batch's metrics.

        Parameters
        ----------
        step : int
            Current step.
        loss : float
            Batch loss.
        lr : float
            Current learning rate.
        batch_time : float
            Batch processing time in seconds.
        gpu_mem_mb : int, optional
            GPU memory usage in MB.
        **kwargs
            Additional metrics.
        """
        metrics = {
            "step": step,
            "phase": "train",
            "loss": loss,
            "learning_rate": lr,
            "batch_time_ms": batch_time * 1000,
        }
        if gpu_mem_mb is not None:
            metrics["gpu_memory_mb"] = gpu_mem_mb
        metrics.update(kwargs)
        self.log(metrics)

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        """
        Log validation metrics.

        Parameters
        ----------
        step : int
            Current step.
        val_loss : float
            Validation loss.
        val_metrics : Dict[str, float]
            Additional validation metrics.
        **kwargs
            Additional metrics.
        """
        metrics = {
            "step": step,
            "phase": "validation",
            "val_loss": val_loss,
        }
        metrics.update(val_metrics)
        metrics.update(kwargs)
        self.log(metrics)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log epoch-level metrics.

        Parameters
        ----------
        epoch : int
            Epoch number.
        metrics : Dict[str, float]
            Epoch metrics.
        """
        metrics["epoch"] = epoch
        metrics["phase"] = "epoch"
        self.log(metrics)

    def close(self) -> None:
        """Close the log file."""
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()


class ExperimentLogger:
    """
    Logger for experiment metadata and summary.
    """

    def __init__(self, log_dir: str | Path, run_id: str):
        """
        Initialize experiment logger.

        Parameters
        ----------
        log_dir : str | Path
            Directory to write logs.
        run_id : str
            Unique run identifier.
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.log_dir / f"manifest_{run_id}.json"
        self.summary_path = self.log_dir / f"summary_{run_id}.json"

    def log_manifest(
        self,
        config: Dict[str, Any],
        git_sha: Optional[str] = None,
        gpu_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write experiment manifest.

        Parameters
        ----------
        config : Dict[str, Any]
            Training configuration.
        git_sha : str, optional
            Git commit SHA.
        gpu_info : Dict[str, Any], optional
            GPU information.
        """
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha or self._get_git_sha(),
            "config": config,
            "gpu_info": gpu_info or self._get_gpu_info(),
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def write_summary(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Write training summary.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Final training metrics.
        """
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _get_git_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU info via nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, check=True,
            )
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "index": int(parts[0]) if len(parts) > 0 else 0,
                    "name": parts[1] if len(parts) > 1 else "unknown",
                    "memory_total_mb": parts[2] if len(parts) > 2 else None,
                    "memory_used_mb": parts[3] if len(parts) > 3 else None,
                    "temperature_c": parts[4] if len(parts) > 4 else None,
                })
            return {"gpus": gpus, "count": len(gpus)}
        except Exception:
            return {"error": "nvidia-smi not available"}


def load_metrics(log_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load all metrics from a log directory.

    Parameters
    ----------
    log_dir : str | Path
        Directory containing metrics.jsonl.

    Returns
    -------
    List[Dict[str, Any]]
        List of logged metrics.
    """
    log_dir = Path(log_dir)
    metrics_path = log_dir / "metrics.jsonl"

    if not metrics_path.exists():
        return []

    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    return metrics
