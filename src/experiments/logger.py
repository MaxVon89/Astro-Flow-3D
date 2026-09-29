#!/usr/bin/env python3
"""
Logging utilities for experiment runs.
Supports JSON-lines metrics, rotating logs, and manifest generation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


class MetricsLogger:
    """Logger for machine-readable metrics in JSON-lines format."""

    def __init__(self, log_dir: str | Path, filename: str = "metrics.jsonl"):
        self.log_dir = Path(log_dir)
        self.metrics_path = self.log_dir / filename
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.metrics_path, "a")

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log a metrics dict as a JSON line."""
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
        """Log a training batch's metrics."""
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
        """Log validation metrics."""
        metrics = {
            "step": step,
            "phase": "validation",
            "val_loss": val_loss,
            **val_metrics,
        }
        metrics.update(kwargs)
        self.log(metrics)

    def close(self) -> None:
        """Close the metrics file."""
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()


class ExperimentLogger:
    """Logger for experiment metadata and stdout/stderr capture."""

    def __init__(self, log_dir: str | Path, run_id: str):
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.stdout_path = self.log_dir / f"stdout_{run_id}.log"
        self.stderr_path = self.log_dir / f"stderr_{run_id}.log"
        self.manifest_path = self.log_dir / f"manifest_{run_id}.json"

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.stdout_file = open(self.stdout_path, "w")
        self.stderr_file = open(self.stderr_path, "w")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / f"python_{run_id}.log"),
                logging.StreamHandler(self._original_stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def redirect_stdout_stderr(self) -> None:
        """Redirect stdout and stderr to log files."""
        sys.stdout = self.stdout_file
        sys.stderr = self.stderr_file

    def restore_stdout_stderr(self) -> None:
        """Restore original stdout and stderr."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def log_manifest(
        self,
        config: Dict[str, Any],
        git_sha: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        gpu_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the experiment manifest."""
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha or self._get_git_sha(),
            "config": config,
            "environment": self._get_env_snapshot(env_vars),
            "gpu_info": gpu_info or self._get_gpu_info(),
            "checkpoint_dir": config.get("checkpointing", {}).get("save_dir", ""),
            "log_dir": str(self.log_dir),
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        self.logger.info(f"Manifest written to {self.manifest_path}")

    def _get_git_sha(self) -> str:
        """Get the current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _get_env_snapshot(self, extra_vars: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Snapshot environment variables."""
        env = {
            "PYTHON_VERSION": sys.version,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        }
        if extra_vars:
            env.update(extra_vars)
        return env

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "name": parts[0] if len(parts) > 0 else "unknown",
                    "memory_total_mb": parts[1] if len(parts) > 1 else None,
                    "memory_used_mb": parts[2] if len(parts) > 2 else None,
                    "temperature_c": parts[3] if len(parts) > 3 else None,
                    "power_draw_w": parts[4] if len(parts) > 4 else None,
                    "power_limit_w": parts[5] if len(parts) > 5 else None,
                })
            return {"gpus": gpus, "count": len(gpus)}
        except Exception as e:
            return {"error": str(e)}

    def close(self) -> None:
        """Close all log files."""
        self.stdout_file.close()
        self.stderr_file.close()
        self.restore_stdout_stderr()


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics (CPU, memory, disk)."""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        "disk_percent": psutil.disk_usage("/").percent,
        "process_count": len(psutil.pids()),
    }


def get_gpu_metrics() -> Dict[str, Any]:
    """Get GPU metrics via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": int(parts[0]) if len(parts) > 0 else 0,
                "name": parts[1] if len(parts) > 1 else "unknown",
                "temperature_c": float(parts[2]) if len(parts) > 2 else 0,
                "power_draw_w": float(parts[3]) if len(parts) > 3 else 0,
                "power_limit_w": float(parts[4]) if len(parts) > 4 else 0,
                "memory_used_mb": float(parts[5]) if len(parts) > 5 else 0,
                "memory_total_mb": float(parts[6]) if len(parts) > 6 else 0,
                "utilization_percent": float(parts[7]) if len(parts) > 7 else 0,
            })
        return {"gpus": gpus, "count": len(gpus)}
    except Exception as e:
        return {"error": str(e)}
