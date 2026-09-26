from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .executor import SafeExecutor
from .planner import RuleBasedPlanner
from .schemas import RunResult, Task
from .validator import ArtifactValidator


class AstroFlowAgent:
    """Minimal local Astro-Flow-3D agent loop: planner -> executor -> validator."""

    def __init__(self, repo_root: str | Path | None = None, dry_run: bool = False):
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        self.planner = RuleBasedPlanner()
        self.executor = SafeExecutor(self.repo_root, dry_run=dry_run)
        self.validator = ArtifactValidator(self.repo_root)

    def run(self, objective: str) -> RunResult:
        tasks = self.planner.plan(objective, self.repo_root)
        if not tasks:
            raise ValueError("Planner produced no tasks for the objective.")

        task = tasks[0]
        result = self.executor.execute(task)
        report = self.validator.validate(task, result)

        run_result = RunResult(
            task_id=task.task_id,
            objective=objective,
            status=report.status,
            artifacts=result.get("artifacts", []),
            metrics=report.metrics,
            validation=report.to_dict(),
            provenance={
                "repo_root": str(self.repo_root),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            error=None if report.status == "pass" else "Validation failed",
        )
        return run_result

    def run_task(self, task: Task) -> RunResult:
        result = self.executor.execute(task)
        report = self.validator.validate(task, result)
        return RunResult(
            task_id=task.task_id,
            objective=task.objective,
            status=report.status,
            artifacts=result.get("artifacts", []),
            metrics=report.metrics,
            validation=report.to_dict(),
            provenance={
                "repo_root": str(self.repo_root),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            error=None if report.status == "pass" else "Validation failed",
        )
