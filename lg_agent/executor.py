from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from lg_agent.schemas import CommandRecord, TaskStep


class SafeExecutor:
    """Safe local execution layer. Only approved Python tools are dispatched."""

    ALLOWED_TOOLS = {
        "repo_context",
        "normalize",
        "generate_tiles",
        "build_dataset",
    }

    def __init__(self, repo_root: str | Path, dry_run: bool = False):
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        self.command_history: List[CommandRecord] = []

    def execute_step(self, step: TaskStep) -> Dict[str, Any]:
        if step.tool not in self.ALLOWED_TOOLS:
            raise PermissionError(f"Tool '{step.tool}' is not allowed by the local executor")

        payload = json.dumps({"tool": step.tool, "args": step.args}, sort_keys=True)
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "from lg_agent.tool_registry import execute_registered_tool; "
                "payload = json.loads(sys.argv[1]); "
                "result = execute_registered_tool(payload['tool'], payload['args']); "
                "print(json.dumps(result, sort_keys=True))"
            ),
            payload,
        ]

        if self.dry_run:
            record = CommandRecord(
                command=" ".join(command[:2] + ["<payload>"]),
                cwd=str(self.repo_root),
                exit_code=0,
                stdout=f"Dry run for tool '{step.tool}'",
                stderr="",
                runtime_seconds=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.command_history.append(record)
            return {"dry_run": True, "step": step.to_dict(), "command": record.command}

        started = datetime.now(timezone.utc)
        run = subprocess.run(command, cwd=str(self.repo_root), capture_output=True, text=True)
        runtime_seconds = (datetime.now(timezone.utc) - started).total_seconds()
        record = CommandRecord(
            command=" ".join(command[:2] + ["<payload>"]),
            cwd=str(self.repo_root),
            exit_code=run.returncode,
            stdout=run.stdout,
            stderr=run.stderr,
            runtime_seconds=runtime_seconds,
            timestamp=started.isoformat(),
        )
        self.command_history.append(record)

        if run.returncode != 0:
            raise RuntimeError(run.stderr or "Tool execution failed")

        try:
            result = json.loads(run.stdout)
        except json.JSONDecodeError:
            result = {"stdout": run.stdout, "stderr": run.stderr, "returncode": run.returncode}

        return {"tool": step.tool, "result": result, "command": record.command}
