from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import Task, ValidationReport


class ArtifactValidator:
    """Simple repo-aware validation for generated task artifacts."""

    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]

    def validate(self, task: Task, result: Dict[str, Any]) -> ValidationReport:
        failures: List[Dict[str, Any]] = []
        artifacts = result.get("artifacts", [])
        observed = [str(item) for item in artifacts]

        metrics: Dict[str, Any] = {
            "expected_outputs": len(task.expected_outputs),
            "observed_outputs": len(observed),
        }

        if not observed:
            failures.append({"rule": "no_artifacts_returned", "tool": task.tool})

        # Validate each expected output
        for expected in task.expected_outputs:
            expected_path = Path(expected)
            matched = False
            
            # Check if any observed artifact matches the expected pattern
            for observed_path in observed:
                obs = Path(observed_path)
                if obs.name == expected_path.name or (expected_path.as_posix() in str(obs)):
                    matched = True
                    break

            # Also check if the actual file exists at expected location  
            candidate = (self.repo_root / expected_path).resolve()
            if candidate.exists():
                matched = True

            metrics[f"output:{expected}"] = matched
            if not matched:
                failures.append({"rule": "required_output_missing", "path": expected})

        # Special validation for dataset building tasks
        if task.tool == "build_dataset":
            manifest_path = self.repo_root / "artifacts" / "dataset" / "manifest.json"
            index_path = self.repo_root / "artifacts" / "dataset" / "index.csv"
            
            if manifest_path.exists():
                metrics["manifest_exists"] = True
                # Basic JSON validation of manifest 
                try:
                    with open(manifest_path, 'r') as f:
                        manifest_data = json.load(f)
                        metrics["manifest_entries"] = len(manifest_data) if isinstance(manifest_data, list) else 1
                except Exception:
                    failures.append({"rule": "manifest_corrupted", "path": str(manifest_path)})
            else:
                failures.append({"rule": "manifest_missing", "path": str(manifest_path)})
                
            if index_path.exists():
                metrics["index_exists"] = True
            else:
                failures.append({"rule": "index_missing", "path": str(index_path)})

        # Special validation for normalization tasks 
        if task.tool == "normalize":
            normalized_path = self.repo_root / "artifacts" / "normalize" / "normalized_image.npy"
            if not normalized_path.exists():
                failures.append({"rule": "normalized_image_missing", "path": str(normalized_path)})

        # Special validation for tile tasks
        if task.tool == "generate_tiles":
            # Check that tiles were created
            tile_dir = self.repo_root / "artifacts" / "tiles"
            if tile_dir.exists():
                tile_count = len(list(tile_dir.glob("*.npy")))
                metrics["tile_count"] = tile_count
                if tile_count == 0:
                    failures.append({"rule": "no_tiles_generated", "path": str(tile_dir)})
            else:
                failures.append({"rule": "tiles_directory_missing", "path": str(tile_dir)})

        status = "pass" if not failures else "fail"
        return ValidationReport(task_id=task.task_id, status=status, metrics=metrics, failures=failures)