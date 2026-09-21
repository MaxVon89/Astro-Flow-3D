"""Typed LangGraph state for one research-to-code cycle."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchDelta(TypedDict):
    method: str
    dataset: str
    claimed_improvement: str
    assumptions: list[str]
    pipeline_delta: str


class PlanTask(TypedDict):
    subtask: str
    hypothesis: str
    validation_criterion: str
    target_files: list[str]
    requires_gpu: bool


class CycleState(TypedDict, total=False):
    research_input: str
    project_status: str
    memory_summary: str
    delta: ResearchDelta
    tasks: list[PlanTask]
    task_index: int
    attempt: int
    max_attempts: int
    experiment_id: str
    workspace_dir: str
    code_result: str
    experiment_result: dict[str, Any]
    validation_passed: bool | None
    validation_notes: str
    gpu_approved: bool
    apply_code: bool
    open_pr: bool
    pr_url: str
    summary: str
    cycle_status: str
    last_error: str
    repo_root: str
    memory_path: str
