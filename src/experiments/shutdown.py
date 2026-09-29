#!/usr/bin/env python3
"""
Controlled shutdown and handoff utilities for training runs.
Handles graceful termination, final checkpoint, and artifact archiving.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

# Run from project root for imports to work
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class ShutdownHandler:
    """Handles graceful shutdown of training processes."""

    def __init__(self, run_id: str, run_timeout_hours: float = 30.0):
        self.run_id = run_id
        self.run_timeout_seconds = run_timeout_hours * 3600
        self.shutdown_initiated = False
        self.process = None

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigint_handler)

    def _sigterm_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM gracefully."""
        self.shutdown_initiated = True
        print(f"\nReceived SIGTERM at {datetime.now(timezone.utc).isoformat()}")
        print("Initiating graceful shutdown...")

    def _sigint_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        self.shutdown_initiated = True
        print(f"\nReceived SIGINT at {datetime.now(timezone.utc).isoformat()}")
        print("Initiating graceful shutdown...")

    def check_timeout(self, start_time: float) -> bool:
        """Check if run has exceeded timeout."""
        elapsed = time.time() - start_time
        return elapsed >= self.run_timeout_seconds

    def is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            process = psutil.Process(pid)
            return process.status() not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def terminate_gracefully(self, pid: int, timeout: int = 30) -> bool:
        """Terminate a process gracefully with SIGTERM, then SIGKILL if needed."""
        try:
            process = psutil.Process(pid)

            # Send SIGTERM first
            process.terminate()
            print(f"Sent SIGTERM to process {pid}")

            # Wait for process to exit
            try:
                process.wait(timeout=timeout)
                print(f"Process {pid} exited gracefully")
                return True
            except psutil.TimeoutExpired:
                print(f"Process {pid} did not exit in time, sending SIGKILL")
                process.kill()
                process.wait(timeout=5)
                print(f"Process {pid} killed")
                return False

        except psutil.NoSuchProcess:
            print(f"Process {pid} no longer exists")
            return True
        except Exception as e:
            print(f"Error terminating process {pid}: {e}")
            return False


class ArtifactHandler:
    """Handles artifact collection and archiving."""

    def __init__(self, artifacts_dir: str | Path, output_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_artifacts(self, run_id: str) -> Dict[str, Any]:
        """Collect all artifacts for a run."""
        artifacts = {
            "run_id": run_id,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "files": [],
            "total_size_mb": 0,
        }

        patterns = [
            ("checkpoints", f"checkpoint_step_*.pt"),
            ("logs", f"*_{run_id}.log"),
            ("logs", f"metrics_{run_id}.jsonl"),
            ("logs", f"manifest_{run_id}.json"),
            ("logs", f"validation_report_{run_id}.json"),
        ]

        for subdir, pattern in patterns:
            dir_path = self.artifacts_dir / subdir
            if dir_path.exists():
                files = list(dir_path.glob(pattern))
                for f in files:
                    artifacts["files"].append({
                        "path": str(f),
                        "name": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "type": subdir,
                    })
                    artifacts["total_size_mb"] += f.stat().st_size / (1024 * 1024)

        return artifacts

    def archive_artifacts(self, run_id: str, output_path: Optional[str] = None) -> Path:
        """Create a tar archive of all artifacts."""
        if output_path is None:
            output_path = self.output_dir / f"artifacts_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.tar.gz"

        output_path = Path(output_path)

        with tarfile.open(output_path, "w:gz") as tar:
            # Add checkpoints
            checkpoints_dir = self.artifacts_dir / "checkpoints"
            if checkpoints_dir.exists():
                for f in checkpoints_dir.glob("checkpoint_*.pt"):
                    tar.add(f, arcname=f"checkpoints/{f.name}")

            # Add logs
            logs_dir = self.artifacts_dir / "logs"
            if logs_dir.exists():
                for f in logs_dir.glob(f"*_{run_id}*"):
                    tar.add(f, arcname=f"logs/{f.name}")

        return output_path

    def write_final_manifest(
        self,
        run_id: str,
        config: Dict[str, Any],
        start_time: float,
        end_time: float,
        checkpoint_path: Optional[str] = None,
    ) -> Path:
        """Write final run manifest with metadata."""
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
            "duration_hours": (end_time - start_time) / 3600,
            "checkpoint_path": checkpoint_path,
            "config": config,
        }

        manifest_path = self.output_dir / f"final_manifest_{run_id}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path


def run_with_timeout(
    command: list,
    run_timeout_hours: float = 30.0,
    checkpoint_freq_minutes: float = 30.0,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a command with timeout and checkpointing."""
    start_time = time.time()
    last_checkpoint_time = start_time
    shutdown_handler = ShutdownHandler("run_with_timeout", run_timeout_hours)
    artifact_handler = ArtifactHandler(
        Path(checkpoint_dir) if checkpoint_dir else Path("/data/astroflow"),
        Path("/data/astroflow/artifacts"),
    )

    result = {
        "success": False,
        "exit_code": None,
        "duration_hours": 0,
        "checkpoint_count": 0,
    }

    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Monitor the process
        while not shutdown_handler.shutdown_initiated:
            # Check timeout
            if shutdown_handler.check_timeout(start_time):
                print(f"Run timeout reached ({run_timeout_hours}h)")
                shutdown_handler.shutdown_initiated = True
                break

            # Check if process is still running
            if process.poll() is not None:
                result["exit_code"] = process.returncode
                break

            # Check if we should checkpoint
            now = time.time()
            if checkpoint_freq_minutes and (now - last_checkpoint_time) >= checkpoint_freq_minutes * 60:
                if checkpoint_dir and Path(checkpoint_dir).exists():
                    checkpoint_count = len(list(Path(checkpoint_dir).glob("checkpoint_*.pt")))
                    print(f"Checkpoint interval reached. Checkpoint count: {checkpoint_count}")
                    last_checkpoint_time = now

            # Poll output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print(line.strip())

            time.sleep(5)

        # Graceful shutdown
        if process.poll() is None:
            shutdown_handler.terminate_gracefully(process.pid, timeout=60)

        result["success"] = True
        result["duration_hours"] = (time.time() - start_time) / 3600

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    return result


def archive_and_notify(
    run_id: str,
    artifacts_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Archive artifacts and prepare handoff."""
    handler = ArtifactHandler(artifacts_dir, output_dir)

    # Collect artifacts
    artifacts = handler.collect_artifacts(run_id)

    # Create archive
    archive_path = handler.archive_artifacts(run_id)

    return {
        "run_id": run_id,
        "archive_path": str(archive_path),
        "artifacts_count": len(artifacts["files"]),
        "total_size_mb": artifacts["total_size_mb"],
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled shutdown and handoff utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Timeout run command
    timeout_parser = subparsers.add_parser("timeout-run", help="Run a command with timeout")
    timeout_parser.add_argument("--command", "-c", nargs="+", required=True, help="Command to run")
    timeout_parser.add_argument("--timeout-hours", "-t", type=float, default=30.0, help="Timeout in hours")
    timeout_parser.add_argument("--checkpoint-freq", "-f", type=float, default=30.0, help="Checkpoint frequency in minutes")
    timeout_parser.add_argument("--checkpoint-dir", "-d", help="Checkpoint directory")

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive artifacts")
    archive_parser.add_argument("--run-id", "-r", required=True, help="Run ID")
    archive_parser.add_argument("--artifacts-dir", "-a", default="/data/astroflow", help="Artifacts directory")
    archive_parser.add_argument("--output-dir", "-o", default="/data/astroflow/artifacts", help="Output directory")

    args = parser.parse_args()

    if args.command == "timeout-run":
        result = run_with_timeout(
            command=args.command,
            run_timeout_hours=args.timeout_hours,
            checkpoint_freq_minutes=args.checkpoint_freq,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "archive":
        result = archive_and_notify(args.run_id, args.artifacts_dir, args.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
