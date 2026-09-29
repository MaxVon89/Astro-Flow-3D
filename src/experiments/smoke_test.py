#!/usr/bin/env python3
"""
Smoke test harness for experiment pre-flight validation.
Validates GPU access, training entrypoint, checkpointing, and logging.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add src directory to path for imports
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


def import_from_path(module_name: str, file_path: Path) -> Any:
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SmokeTestRunner:
    """Runs smoke tests for experiment validation."""

    def __init__(self, run_id: str, artifacts_dir: str | Path):
        self.run_id = run_id
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "overall_status": "pending",
        }

    def test_gpu_access(self) -> Dict[str, Any]:
        """Test that GPUs are accessible."""
        status = "pass"
        details = {}

        try:
            device_count = torch.cuda.device_count()
            details["device_count"] = device_count

            if device_count == 0:
                status = "fail"
                details["error"] = "No GPUs detected"
            else:
                for i in range(device_count):
                    details[f"gpu_{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_memory_mb": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024),
                    }

                    # Test basic CUDA operation
                    x = torch.randn(100, 100).cuda(i)
                    y = x @ x.T
                    y_sum = y.sum()
                    if torch.isnan(y_sum).item():
                        status = "fail"
                        details[f"gpu_{i}"]["matmul_test"] = "failed (NaN)"
                    else:
                        details[f"gpu_{i}"]["matmul_test"] = "passed"

        except Exception as e:
            status = "fail"
            details["error"] = str(e)

        self.results["tests"]["gpu_access"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["gpu_access"]

    def test_python_imports(self) -> Dict[str, Any]:
        """Test that required Python modules can be imported."""
        status = "pass"
        details = {}

        required_modules = [
            "torch",
            "numpy",
            "yaml",
            "psutil",
        ]

        for module in required_modules:
            try:
                __import__(module)
                details[module] = "imported"
            except ImportError as e:
                status = "fail"
                details[module] = f"import failed: {e}"

        self.results["tests"]["python_imports"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["python_imports"]

    def test_entrypoint(self) -> Dict[str, Any]:
        """Test that the training entrypoint runs (dry-run)."""
        status = "pass"
        details = {}

        try:
            # Import modules directly from path using _src_path defined at module level
            run_py = _src_path / "run.py"
            checkpoint_py = _src_path / "checkpoint.py"

            run_module = import_from_path("smoke_run_module", run_py)
            checkpoint_module = import_from_path("smoke_checkpoint_module", checkpoint_py)

            # Get classes
            TrainingRunner = getattr(run_module, "TrainingRunner")
            CheckpointManager = getattr(checkpoint_module, "CheckpointManager")

            details["runner_import"] = "success"

            # Create a minimal config
            config = {
                "run": {"id": f"smoke_test_{self.run_id}"},
                "data": {"batch_size": 2, "num_workers": 1},
                "training": {"steps": 1, "learning_rate": 1e-4},
                "checkpointing": {"save_dir": str(self.artifacts_dir / "checkpoints"), "enabled": True},
                "logging": {"log_dir": str(self.artifacts_dir / "logs")},
                "resources": {"visible_devices": "0"},
            }

            runner = TrainingRunner(config)

            # Test model creation
            model, optimizer = runner._load_model_and_optimizer()
            details["model_creation"] = "success"
            details["device"] = str(runner.device)

            # Test forward pass
            batch = torch.randn(1, 4, 256, 256).to(runner.device)
            output = model(batch)
            details["forward_pass"] = "success"
            details["output_shape"] = str(output.shape)

        except Exception as e:
            status = "fail"
            details["error"] = str(e)
            import traceback
            details["traceback"] = traceback.format_exc()

        self.results["tests"]["entrypoint"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["entrypoint"]

    def test_checkpointing(self) -> Dict[str, Any]:
        """Test checkpoint save and load."""
        status = "pass"
        details = {}

        try:
            # Import modules directly from path using _src_path defined at module level
            checkpoint_py = _src_path / "checkpoint.py"
            run_py = _src_path / "run.py"

            checkpoint_module = import_from_path("smoke_checkpoint_module", checkpoint_py)
            run_module = import_from_path("smoke_run_module", run_py)

            CheckpointManager = getattr(checkpoint_module, "CheckpointManager")
            TrainingRunner = getattr(run_module, "TrainingRunner")
            from torch import nn

            # Setup
            config = {
                "checkpointing": {"save_dir": str(self.artifacts_dir / "checkpoints"), "enabled": True},
            }
            manager = TrainingRunner(config).checkpoint_manager
            manager.save_dir = self.artifacts_dir / "checkpoints"

            # Create test model
            model = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.ReLU()).to("cuda:0")
            optimizer = torch.optim.Adam(model.parameters())

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                model=model,
                step=1,
                epoch=0,
                optimizer=optimizer,
                metadata={"test": "smoke_test"},
            )
            details["save_checkpoint"] = "success"
            details["checkpoint_path"] = str(checkpoint_path)

            # Load checkpoint
            resume_info = manager.resume_from_checkpoint(model, optimizer)
            details["load_checkpoint"] = "success"
            details["resume_info"] = resume_info

            # Verify checkpoint exists
            if not checkpoint_path.exists():
                status = "fail"
                details["error"] = "Checkpoint file not found after save"

        except Exception as e:
            status = "fail"
            details["error"] = str(e)
            import traceback
            details["traceback"] = traceback.format_exc()

        self.results["tests"]["checkpointing"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["checkpointing"]

    def test_logging(self) -> Dict[str, Any]:
        """Test logging functionality."""
        status = "pass"
        details = {}

        try:
            # Import modules directly from path using _src_path defined at module level
            logger_py = _src_path / "logger.py"
            logger_module = import_from_path("smoke_logger_module", logger_py)

            MetricsLogger = getattr(logger_module, "MetricsLogger")
            ExperimentLogger = getattr(logger_module, "ExperimentLogger")

            # Test MetricsLogger
            metrics_logger = MetricsLogger(self.artifacts_dir / "logs", "smoke_test_metrics.jsonl")
            metrics_logger.log({"test": "smoke_test", "step": 1, "loss": 0.5})
            metrics_logger.close()
            details["metrics_logger"] = "success"

            # Test ExperimentLogger
            exp_logger = ExperimentLogger(self.artifacts_dir / "logs", f"smoke_{self.run_id}")
            exp_logger.logger.info("Smoke test log entry")
            exp_logger.close()
            details["experiment_logger"] = "success"

            # Verify files were created
            if not (self.artifacts_dir / "logs" / "smoke_test_metrics.jsonl").exists():
                status = "fail"
                details["error"] = "Metrics file not created"

            if not (self.artifacts_dir / "logs" / f"python_smoke_{self.run_id}.log").exists():
                status = "fail"
                details["error"] = "Log file not created"

        except Exception as e:
            status = "fail"
            details["error"] = str(e)

        self.results["tests"]["logging"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["logging"]

    def test_dataset_access(self) -> Dict[str, Any]:
        """Test dataset paths are accessible."""
        status = "pass"
        details = {}

        # Check for dataset paths from config
        expected_paths = [
            "/data/astroflow/datasets/ceers_v1/train",
            "/data/astroflow/datasets/ceers_v1/val",
        ]

        for path in expected_paths:
            p = Path(path)
            if p.exists():
                # Check for at least one .npy file
                npy_files = list(p.glob("*.npy"))
                details[path] = {
                    "exists": True,
                    "npy_files": len(npy_files),
                }
            else:
                # This is expected if dataset hasn't been built yet
                details[path] = {
                    "exists": False,
                    "note": "Dataset not built yet - this is expected before training",
                }

        self.results["tests"]["dataset_access"] = {
            "status": status,
            "details": details,
        }
        return self.results["tests"]["dataset_access"]

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all smoke tests."""
        self.test_gpu_access()
        self.test_python_imports()
        self.test_entrypoint()
        self.test_checkpointing()
        self.test_logging()
        self.test_dataset_access()

        # Determine overall status
        statuses = [t["status"] for t in self.results["tests"].values()]
        if all(s == "pass" for s in statuses):
            self.results["overall_status"] = "pass"
        elif any(s == "fail" for s in statuses):
            self.results["overall_status"] = "fail"
        else:
            self.results["overall_status"] = "warning"

        return self.results

    def write_results(self, output_path: Optional[str] = None) -> Path:
        """Write results to JSON file."""
        if output_path is None:
            output_path = self.artifacts_dir / f"smoke_test_results_{self.run_id}.json"

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return output_path


def run_smoke_tests(run_id: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run smoke tests and return results."""
    if run_id is None:
        run_id = f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if output_dir is None:
        output_dir = "/data/astroflow/artifacts"

    runner = SmokeTestRunner(run_id, output_dir)
    results = runner.run_all_tests()
    runner.write_results()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests for experiment validation")
    parser.add_argument("--run-id", "-r", help="Run ID (generated if not provided)")
    parser.add_argument("--output-dir", "-o", default="/data/astroflow/artifacts", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output only JSON")
    args = parser.parse_args()

    results = run_smoke_tests(args.run_id, args.output_dir)

    if not args.json:
        print("=" * 60)
        print("SMOKE TEST RESULTS")
        print("=" * 60)
        print(f"Run ID: {results['run_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        print()

        for name, test in results["tests"].items():
            status = "✓" if test["status"] == "pass" else "✗"
            print(f"{status} {name}: {test['status']}")
            if test["status"] == "fail" and "details" in test:
                for key, value in test["details"].items():
                    if key != "traceback":
                        print(f"    {key}: {value}")

        print()
        print(f"Results written to: {results['run_id']}")
    else:
        print(json.dumps(results, indent=2))

    # Exit with error code if any test failed
    if results["overall_status"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
