"""Layer 2: science-aware planner with explicit validation criteria."""

from __future__ import annotations

from .llm import LLMClient, extract_json
from .prompts import PLANNER_SYSTEM
from .state import CycleState, PlanTask


def plan_tasks(llm: LLMClient, state: CycleState) -> list[PlanTask]:
    delta = state.get("delta") or {}
    user = (
        f"RESEARCH DELTA:\n{delta}\n\n"
        f"PROJECT STATUS:\n{state.get('project_status', '')}\n\n"
        f"EXPERIMENT MEMORY:\n{state.get('memory_summary', '')}\n\n"
        f"previous_attempt_error: {state.get('last_error', '')}\n"
        f"attempt: {state.get('attempt', 0)}"
    )
    parsed = extract_json(llm.complete(PLANNER_SYSTEM, user))
    tasks: list[PlanTask] = []
    for item in parsed.get("tasks") or []:
        tasks.append(
            PlanTask(
                subtask=str(item.get("subtask", "")),
                hypothesis=str(item.get("hypothesis", "")),
                validation_criterion=str(item.get("validation_criterion", "")),
                target_files=[str(path) for path in item.get("target_files") or []],
                requires_gpu=bool(item.get("requires_gpu", False)),
            )
        )
    if not tasks:
        raise ValueError("Planner returned no tasks")
    return tasks
