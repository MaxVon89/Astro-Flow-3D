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

        for expected in task.expected_outputs:
            expected_path = Path(expected)
            matched = False
            for observed_path in observed:
                obs = Path(observed_path)
                if obs.name == expected_path.name or (Path(expected_path).as_posix() in str(obs)):
                    matched = True
                    break

            candidate = (self.repo_root / expected_path).resolve()
            if candidate.exists():
                matched = True

            metrics[f"output:{expected}"] = matched
            if not matched:
                failures.append({"rule": "required_output_missing", "path": expected})

        if task.tool == "build_dataset":
            manifest_path = self.repo_root / "artifacts" / "dataset" / "manifest.json"
            index_path = self.repo_root / "artifacts" / "dataset" / "index.csv"
            if manifest_path.exists():
                metrics["manifest_exists"] = True
            else:
                failures.append({"rule": "manifest_missing", "path": str(manifest_path)})
            if index_path.exists():
                metrics["index_exists"] = True
            else:
                failures.append({"rule": "index_missing", "path": str(index_path)})

        status = "pass" if not failures else "fail"
        return ValidationReport(task_id=task.task_id, status=status, metrics=metrics, failures=failures)
