"""Production-oriented LangGraph runtime for Astro-Flow-3D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lg_agent.graph import AstroFlowMultiAgent


def run_cli(objective: str, repo_root: str, dry_run: bool = False) -> Dict[str, Any]:
    agent = AstroFlowMultiAgent(repo_root=repo_root, dry_run=dry_run)
    state = agent.run_task(objective=objective, user_request=objective)
    plan = []
    for step in state.get("plan", []):
        if hasattr(step, "to_dict"):
            plan.append(step.to_dict())
        else:
            plan.append(step)
    return {
        "status": state.get("status"),
        "task_id": state.get("task_id"),
        "objective": state.get("objective"),
        "plan": plan,
        "artifacts": state.get("artifact_paths", []),
        "validation": None if state.get("validation_report") is None else state["validation_report"].to_dict(),
        "run_manifest": None if state.get("run_manifest") is None else state["run_manifest"].to_dict(),
        "error": state.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Astro-Flow-3D local LangGraph agent.")
    parser.add_argument("--objective", required=True, help="Mission objective for the agent to execute.")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="Project root to inspect and operate on.")
    parser.add_argument("--dry-run", action="store_true", help="Run the orchestration without mutating repo state.")
    args = parser.parse_args()

    result = run_cli(args.objective, args.repo_root, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
