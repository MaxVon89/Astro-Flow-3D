#!/usr/bin/env python3
"""
Experiment runner for persistent GPU training.
Supports checkpointing, resume, monitoring, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


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
        """Find the most recent checkpoint file by step number."""
        if not self.enabled:
            return None

        checkpoint_files = list(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))

        if not checkpoint_files:
            return None

        # Sort by numeric step value, not lexicographically
        def extract_step(path: Path) -> int:
            stem = path.stem
            if "step_" in stem:
                return int(stem.split("step_")[-1])
            elif "epoch_" in stem:
                return int(stem.split("epoch_")[-1])
            return 0

        return sorted(checkpoint_files, key=extract_step)[-1]

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

    def validate_checkpoint(self, checkpoint_path: Path) -> tuple[bool, str]:
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
            import subprocess
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
            import subprocess
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
    import psutil
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
        import subprocess
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


import logging


class TrainingRunner:
    """Manages a training run with checkpointing, monitoring, and graceful shutdown."""

    def __init__(
        self,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
    ):
        self.config = config
        self.run_id = run_id or config.get("run", {}).get("id", "unknown")
        self.running = False

        # Extract config sections
        self.run_config = config.get("run", {})
        self.data_config = config.get("data", {})
        self.training_config = config.get("training", {})
        self.checkpoint_config = config.get("checkpointing", {})
        self.logging_config = config.get("logging", {})
        self.validation_config = config.get("validation", {})
        self.resource_config = config.get("resources", {})

        # Initialize components
        self.log_dir = Path(self.logging_config.get("log_dir", "/data/astroflow/logs"))
        self.checkpoint_dir = Path(self.checkpoint_config.get("save_dir", "/data/astroflow/checkpoints"))

        # Set up logging
        self.logger = ExperimentLogger(self.log_dir, self.run_id)
        self.metrics_logger = MetricsLogger(self.log_dir, "metrics.jsonl")

        # Set up checkpointing
        self.checkpoint_manager = create_checkpoint_manager_from_config(config)

        # Tracking state
        self.start_time = None
        self.current_step = 0
        self.current_epoch = 0
        self.best_metric = None

        # Set CUDA devices
        self._setup_devices()

    def _setup_devices(self) -> None:
        """Configure CUDA_VISIBLE_DEVICES and device selection."""
        visible_devices = self.resource_config.get("visible_devices", "0,1,2,3,4,5,6,7")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Check NVIDIA drivers.")

        device_count = torch.cuda.device_count()
        self.logger.logger.info(f"Found {device_count} GPU(s)")

        if device_count > 1 and self.resource_config.get("use_ddp", False):
            self.device = "cuda"
            self.use_ddp = True
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.use_ddp = False

        self.logger.logger.info(f"Using device: {self.device}")

    def _load_model_and_optimizer(self) -> tuple:
        """Load or create model and optimizer."""
        from torch import nn

        model = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        ).to(self.device)

        if self.use_ddp:
            model = nn.parallel.DistributedDataParallel(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 1e-5)),
        )

        return model, optimizer

    def _create_dataloader(self) -> Any:
        """Create training dataloader."""
        from torch.utils.data import DataLoader, TensorDataset

        batch_size = self.data_config.get("batch_size", 16)
        num_workers = self.data_config.get("num_workers", 8)

        dummy_data = torch.randn(1000, 4, 256, 256)
        dataset = TensorDataset(dummy_data)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def _training_step(self, model: Any, batch: Any, optimizer: Any) -> float:
        """Execute a single training step."""
        model.train()
        optimizer.zero_grad()
        output = model(batch[0].to(self.device))
        loss = output.mean()

        loss.backward()
        optimizer.step()

        return loss.item()

    def _validate(self, model: Any) -> Dict[str, float]:
        """Run validation and return metrics."""
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(10):
                batch = torch.randn(2, 4, 256, 256).to(self.device)
                output = model(batch)
                val_loss += output.mean().item()

            val_loss /= 10

        # Return only float values (no complex numbers)
        return {"val_loss": float(val_loss), "val_rmse": float(abs(val_loss) ** 0.5)}

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run the training loop."""
        self.running = True
        self.start_time = time.time()

        self.logger.logger.info(f"Starting training run: {self.run_id}")
        self.logger.log_manifest(self.config)

        model, optimizer = self._load_model_and_optimizer()

        resume_info = self.checkpoint_manager.resume_from_checkpoint(
            model, optimizer, map_location=self.device
        )

        self.current_step = resume_info.get("step", 0)
        self.current_epoch = resume_info.get("epoch", 0)

        self.logger.logger.info(f"Resumed from step {self.current_step}, epoch {self.current_epoch}")

        dataloader = self._create_dataloader()

        steps_per_epoch = len(dataloader)
        max_steps = max_steps or self.training_config.get("steps", 50000)
        save_frequency_minutes = self.checkpoint_config.get("save_frequency_minutes", 30)
        val_interval_steps = self.validation_config.get("interval_steps", 500)

        last_checkpoint_time = time.time()
        last_log_time = time.time()

        try:
            while self.running and self.current_step < max_steps:
                for batch in dataloader:
                    if not self.running or self.current_step >= max_steps:
                        break

                    batch_start = time.time()
                    loss = self._training_step(model, batch, optimizer)
                    batch_time = time.time() - batch_start

                    current_lr = optimizer.param_groups[0]["lr"]
                    gpu_metrics = get_gpu_metrics()
                    gpu_mem = gpu_metrics.get("gpus", [{}])[0].get("memory_used_mb", 0)

                    self.metrics_logger.log_batch(
                        step=self.current_step,
                        loss=loss,
                        lr=current_lr,
                        batch_time=batch_time,
                        gpu_mem_mb=gpu_mem,
                    )

                    self.current_step += 1

                    now = time.time()
                    if now - last_checkpoint_time >= save_frequency_minutes * 60:
                        self._save_checkpoint(model, optimizer)
                        last_checkpoint_time = now

                    if self.current_step % val_interval_steps == 0:
                        val_metrics = self._validate(model)
                        self.metrics_logger.log_validation(
                            step=self.current_step,
                            val_loss=val_metrics["val_loss"],
                            val_metrics=val_metrics,
                        )
                        self.logger.logger.info(
                            f"Step {self.current_step} | "
                            f"Loss: {loss:.4f} | "
                            f"Val Loss: {val_metrics['val_loss']:.4f}"
                        )

                    if now - last_log_time >= 60:
                        system_metrics = get_system_metrics()
                        self.logger.logger.info(
                            f"Step {self.current_step} | "
                            f"CPU: {system_metrics['cpu_percent']:.1f}% | "
                            f"RAM: {system_metrics['memory_used_mb']:.0f}MB"
                        )
                        last_log_time = now

                    elapsed = (time.time() - self.start_time) / 3600
                    self.logger.logger.info(
                        f"Progress: Step {self.current_step}/{max_steps} | "
                        f"Time: {elapsed:.1f}h"
                    )

        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
            self.running = False

        self._save_checkpoint(model, optimizer, final=True)
        self.metrics_logger.close()
        self.logger.close()

        return {
            "status": "completed",
            "steps": self.current_step,
            "epochs": self.current_epoch,
            "duration_hours": (time.time() - self.start_time) / 3600,
        }

    def _save_checkpoint(self, model: Any, optimizer: Any, final: bool = False) -> Path:
        """Save a checkpoint."""
        return self.checkpoint_manager.save_checkpoint(
            model=model,
            step=self.current_step,
            epoch=self.current_epoch,
            optimizer=optimizer,
            best_metric=self.best_metric,
            metadata={"final": final, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

    def shutdown(self) -> None:
        """Gracefully shutdown training."""
        self.logger.logger.info("Initiating graceful shutdown...")
        self.running = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def main():
    parser = argparse.ArgumentParser(description="Run persistent training experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    parser.add_argument("--steps", "-s", type=int, help="Override max steps")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually train")
    args = parser.parse_args()

    config = load_config(args.config)
    config["run"]["id"] = create_run_id()

    if args.dry_run:
        print(f"Would run: {config['run']['id']}")
        print(f"Steps: {args.steps or config['training'].get('steps', 50000)}")
        return

    runner = TrainingRunner(config)
    result = runner.run(max_steps=args.steps)
    print(f"Training complete: {result}")


if __name__ == "__main__":
    main()
