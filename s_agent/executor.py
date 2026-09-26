from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from lg_agent.tool_registry import execute_registered_tool

from .schemas import Task


class SafeExecutor:
    """Minimal allowlisted local executor for repo-aware tasks."""

    ALLOWED_TOOLS = {"repo_context", "normalize", "generate_tiles", "build_dataset"}

    def __init__(self, repo_root: str | Path | None = None, dry_run: bool = False):
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        self.dry_run = dry_run
        self.history: List[Dict[str, Any]] = []

    def execute(self, task: Task) -> Dict[str, Any]:
        if task.tool not in self.ALLOWED_TOOLS:
            raise PermissionError(f"Tool '{task.tool}' is not allowed by the local executor.")

        if self.dry_run:
            result = {
                "task_id": task.task_id,
                "tool": task.tool,
                "status": "dry_run",
                "artifacts": [],
                "metrics": {"dry_run": True},
            }
            self.history.append({"task_id": task.task_id, "tool": task.tool, "dry_run": True})
            return result

        payload = dict(task.args)
        payload.setdefault("repo_root", str(self.repo_root))
        result = execute_registered_tool(task.tool, payload)
        self.history.append({"task_id": task.task_id, "tool": task.tool, "result": result})
        return result
