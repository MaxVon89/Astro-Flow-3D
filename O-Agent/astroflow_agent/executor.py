"""Layer 3a–b: write proposed code artifacts, then run (or refuse) experiments."""

from __future__ import annotations

from pathlib import Path

from .compute import ComputeBackend, LocalHardcodedBackend
from .science import flux_ratio_report
from .state import CycleState, PlanTask


def _current_task(state: CycleState) -> PlanTask:
    tasks = state["tasks"]
    return tasks[state.get("task_index", 0)]


def write_code_artifacts(state: CycleState, repo_root: Path) -> dict:
    task = _current_task(state)
    experiment_id = state["experiment_id"]
    workspace = Path(state.get("workspace_dir") or (repo_root / "Agent" / "workspace" / experiment_id))
    workspace.mkdir(parents=True, exist_ok=True)
    plan_path = workspace / "PLAN.md"
    plan_path.write_text(
        "\n".join(
            [
                f"# {task['subtask']}",
                "",
                f"Hypothesis: {task['hypothesis']}",
                "",
                f"Validation: {task['validation_criterion']}",
                "",
                "Target files:",
                *[f"- {path}" for path in task["target_files"]],
                "",
                "Do not push to main. Open a PR after scientific validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    snippets = []
    for relative in task["target_files"]:
        source = repo_root / relative
        if source.is_file():
            snippets.append(f"## {relative}\n\n```\n{source.read_text(encoding='utf-8')[:4000]}\n```")
    if snippets:
        (workspace / "CONTEXT.md").write_text("\n\n".join(snippets), encoding="utf-8")
    return {
        "workspace": str(workspace),
        "plan_path": str(plan_path),
        "applied_to_working_tree": False,
        "note": "Code lands in Agent/workspace for PR review; the graph never pushes main.",
    }


def run_experiment(state: CycleState, backend: ComputeBackend) -> dict:
    task = _current_task(state)
    if task["requires_gpu"] and not state.get("gpu_approved"):
        return {
            "status": "blocked",
            "reason": "GPU job requires --approve-gpu human checkpoint",
            "backend": backend.name,
        }
    if task["requires_gpu"] and not isinstance(backend, LocalHardcodedBackend):
        handle = backend.submit(
            {
                "experiment_id": state["experiment_id"],
                "subtask": task["subtask"],
                "command": "remote train/eval",
            }
        )
        handle = backend.poll(handle)
        return {"status": handle.status, "backend": handle.backend, "result": handle.result}

    criterion = task["validation_criterion"].lower() + " " + task["subtask"].lower()
    payload: dict = {"status": "completed", "backend": backend.name}
    if any(token in criterion for token in ("flux", "ratio", "asinh", "normaliz")):
        payload["flux_ratio"] = flux_ratio_report()
    handle = backend.submit({"experiment_id": state["experiment_id"], "command": "local scientific probe"})
    payload["job"] = {"job_id": handle.job_id, "status": handle.status}
    return payload
