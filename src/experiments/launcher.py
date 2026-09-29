#!/usr/bin/env python3
"""
Astro-Flow-3D Experiment Launcher

Orchestrates the 30-hour persistent GPU training run:
1. Runs smoke tests (pre-flight validation)
2. Starts the training runner with checkpointing
3. Monitors GPU health during the run
4. Handles graceful shutdown and artifact archiving
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def run_smoke_tests(config_path: str) -> bool:
    """Run smoke tests to validate the setup."""
    print("=" * 60)
    print("SMOKE TESTS (Pre-flight Validation)")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "src/experiments/smoke_test.py"],
        capture_output=True,
        text=True,
    )

    # The smoke test prints formatted output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("SMOKE TESTS FAILED!")
        return False

    return True


def start_training(config_path: str, steps: int | None) -> subprocess.Popen:
    """Start the training runner."""
    cmd = [sys.executable, "src/experiments/run.py", "--config", config_path]

    if steps:
        cmd.extend(["--steps", str(steps)])

    print(f"Starting training: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_gpu_monitor(interval: int = 60) -> subprocess.Popen | None:
    """Start GPU monitoring in the background."""
    try:
        cmd = [sys.executable, "scripts/gpu_monitor.py", "--interval", str(interval)]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        print(f"Warning: Could not start GPU monitor: {e}")
        return None


def run_experiment(
    config_path: str,
    duration_hours: float = 30.0,
    steps: int | None = None,
    monitor: bool = True,
) -> dict:
    """Run the complete experiment."""
    start_time = time.time()

    print("=" * 60)
    print("ASTRO-FLOW-3D EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Duration: {duration_hours} hours")
    print(f"Start Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Step 1: Run smoke tests
    if not run_smoke_tests(config_path):
        return {"status": "failed", "reason": "Smoke tests failed"}

    # Step 2: Start training
    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    training_proc = start_training(config_path, steps)

    # Step 3: Start GPU monitor (optional)
    monitor_proc = None
    if monitor:
        print()
        print("Starting GPU monitor...")
        monitor_proc = start_gpu_monitor(interval=60)

    # Step 4: Monitor until timeout
    timeout_seconds = duration_hours * 3600
    try:
        while time.time() - start_time < timeout_seconds:
            # Check if training is still running
            if training_proc.poll() is not None:
                print(f"Training finished with exit code {training_proc.returncode}")
                break

            # Check GPU status
            if monitor_proc and monitor_proc.poll() is not None:
                print("GPU monitor exited")

            # Print progress every 10 minutes
            elapsed = (time.time() - start_time) / 60
            remaining = (timeout_seconds - (time.time() - start_time)) / 60
            print(f"Elapsed: {elapsed:.1f} min | Remaining: {remaining:.1f} min")

            # Read and display output
            if training_proc.stdout:
                line = training_proc.stdout.readline()
                if line:
                    print(line.strip())

            time.sleep(60)  # Poll every minute

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print()
        print("=" * 60)
        print("SHUTDOWN")
        print("=" * 60)

        # Terminate training
        if training_proc.poll() is None:
            print("Terminating training process...")
            training_proc.terminate()
            try:
                training_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training_proc.kill()

        # Terminate monitor
        if monitor_proc and monitor_proc.poll() is None:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()

        # Calculate final stats
        duration = (time.time() - start_time) / 3600
        return {
            "status": "completed",
            "duration_hours": duration,
            "exit_code": training_proc.returncode,
            "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Astro-Flow-3D Experiment Launcher"
    )
    parser.add_argument(
        "--config", "-c", default="configs/experiments/persistent_run.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=30.0,
        help="Target duration in hours (default: 30)"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None,
        help="Override max steps"
    )
    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Disable GPU monitoring"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("Dry run - would execute:")
        print(f"  Config: {args.config}")
        print(f"  Duration: {args.duration} hours")
        if args.steps:
            print(f"  Steps: {args.steps}")
        if not args.no_monitor:
            print("  GPU monitoring: enabled")
        return

    result = run_experiment(
        config_path=args.config,
        duration_hours=args.duration,
        steps=args.steps,
        monitor=not args.no_monitor,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
