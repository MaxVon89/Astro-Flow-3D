from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from lg_agent.schemas import ValidationReport


class OutputValidator:
    """Repository-aware validator for produced artifacts."""

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root)

    def validate(self, task_id: str, outputs: Dict[str, Any], expected_outputs: List[str]) -> ValidationReport:
        failures: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {
            "expected_outputs": len(expected_outputs),
            "observed_outputs": 0,
        }

        observed = []
        for key, value in outputs.items():
            if isinstance(value, str):
                observed.append(value)
            elif isinstance(value, list):
                observed.extend(str(item) for item in value)

        observed_names = {Path(item).name for item in observed if isinstance(item, str)}

        for path in expected_outputs:
            print_path = Path(path)
            candidates = [
                self.repo_root / print_path if not os.path.isabs(str(print_path)) else print_path,
                self.repo_root / print_path.name if not os.path.isabs(str(print_path)) else print_path,
            ]
            exists = any(candidate.exists() for candidate in candidates)
            if not exists and print_path.name in observed_names:
                exists = True
            metrics[f"output:{path}"] = exists
            if not exists:
                failures.append({"rule": "required_output_missing", "path": path})

        for artifact in observed:
            path = Path(artifact)
            if path.exists():
                metrics["observed_outputs"] += 1

        if failures:
            return ValidationReport(task_id=task_id, status="fail", metrics=metrics, failures=failures)

        return ValidationReport(task_id=task_id, status="pass", metrics={**metrics, "validated": True}, failures=[])
