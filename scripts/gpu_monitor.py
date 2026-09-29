#!/usr/bin/env python3
"""
GPU health monitoring script.
Monitors GPU utilization, temperature, memory usage and logs alerts.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class GPUMonitor:
    """Monitor GPU health and metrics."""

    def __init__(self, log_dir: Optional[str] = None, threshold_util: float = 20.0):
        self.log_dir = Path(log_dir) if log_dir else Path("/data/astroflow/logs")
        self.threshold_util = threshold_util
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "gpu_health.jsonl"

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Query nvidia-smi for current GPU metrics."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,"
                    "memory.used,memory.total,utilization.gpu",
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
                    "encoder_util_percent": float(parts[8]) if len(parts) > 8 else 0,
                })
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "count": len(gpus),
                "gpus": gpus,
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"nvidia-smi failed: {e}", "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    def check_health(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check GPU health and return any alerts."""
        alerts = []
        gpus = metrics.get("gpus", [])

        for gpu in gpus:
            # Check temperature (alert if > 85C)
            if gpu.get("temperature_c", 0) > 85:
                alerts.append({
                    "level": "critical",
                    "gpu": gpu.get("index"),
                    "metric": "temperature",
                    "value": gpu.get("temperature_c"),
                    "threshold": 85,
                    "message": f"GPU {gpu.get('index')} temperature critical: {gpu.get('temperature_c')}C",
                })

            # Check utilization (alert if consistently < threshold during training)
            if gpu.get("utilization_percent", 100) < self.threshold_util:
                alerts.append({
                    "level": "warning",
                    "gpu": gpu.get("index"),
                    "metric": "utilization",
                    "value": gpu.get("utilization_percent"),
                    "threshold": self.threshold_util,
                    "message": f"GPU {gpu.get('index')} utilization below threshold: {gpu.get('utilization_percent')}%",
                })

            # Check power draw
            power_draw = gpu.get("power_draw_w", 0)
            power_limit = gpu.get("power_limit_w", 0)
            if power_limit > 0 and power_draw / power_limit > 0.95:
                alerts.append({
                    "level": "warning",
                    "gpu": gpu.get("index"),
                    "metric": "power",
                    "value": power_draw,
                    "threshold": power_limit * 0.95,
                    "message": f"GPU {gpu.get('index')} power draw near limit: {power_draw}W / {power_limit}W",
                })

        return alerts

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to file."""
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def run_monitoring(self, interval: int = 60, duration: Optional[int] = None) -> None:
        """Run continuous monitoring loop."""
        start_time = time.time()
        interval_count = 0

        print(f"Starting GPU monitoring (interval={interval}s)...")

        try:
            while True:
                metrics = self.get_gpu_metrics()
                alerts = self.check_health(metrics)

                # Log metrics
                self.log_metrics(metrics)

                # Print summary
                interval_count += 1
                print(f"[{interval_count}] {metrics.get('timestamp', 'N/A')}")
                print(f"  GPUs: {metrics.get('count', 0)}")

                for gpu in metrics.get("gpus", []):
                    util = gpu.get("utilization_percent", 0)
                    temp = gpu.get("temperature_c", 0)
                    mem = gpu.get("memory_used_mb", 0)
                    print(f"    GPU {gpu.get('index')}: {util}% util, {temp}C, {mem:.0f}MB mem")

                # Print alerts
                if alerts:
                    print("  ALERTS:")
                    for alert in alerts:
                        print(f"    [{alert['level'].upper()}] {alert['message']}")

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print("Duration exceeded, stopping.")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU health")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--duration", "-d", type=int, help="Duration to run in seconds")
    parser.add_argument("--threshold-util", "-t", type=float, default=20.0, help="Low utilization threshold %")
    parser.add_argument("--log-dir", "-l", help="Directory for log files")
    parser.add_argument("--single", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    monitor = GPUMonitor(log_dir=args.log_dir, threshold_util=args.threshold_util)

    if args.single:
        metrics = monitor.get_gpu_metrics()
        alerts = monitor.check_health(metrics)
        print(json.dumps({"metrics": metrics, "alerts": alerts}, indent=2))
    else:
        monitor.run_monitoring(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()
