"""LangGraph research-to-code loop: intake → plan → execute → validate → memory → PR."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .compute import ComputeBackend, LocalHardcodedBackend
from .executor import run_experiment, write_code_artifacts
from .github_pr import propose_pull_request
from .intake import intake_research, load_project_status
from .llm import LLMClient
from .memory import ExperimentRecord, ExperimentStore
from .planner import plan_tasks
from .state import CycleState
from .validator import validate_science


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        store: ExperimentStore,
        repo_root: Path,
        backend: ComputeBackend | None = None,
    ):
        self.llm = llm
        self.store = store
        self.repo_root = Path(repo_root)
        self.backend = backend or LocalHardcodedBackend()

    def intake(self, state: CycleState) -> dict:
        status = load_project_status(self.repo_root)
        memory_summary = self.store.summary()
        delta = intake_research(
            self.llm,
            {
                **state,
                "project_status": status,
                "memory_summary": memory_summary,
            },
        )
        return {
            "project_status": status,
            "memory_summary": memory_summary,
            "delta": delta,
            "task_index": 0,
            "attempt": 0,
            "cycle_status": "running",
            "repo_root": str(self.repo_root),
            "memory_path": str(self.store.path),
        }

    def planner(self, state: CycleState) -> dict:
        tasks = plan_tasks(self.llm, state)
        experiment_id = uuid4().hex[:12]
        workspace = self.repo_root / "Agent" / "workspace" / experiment_id
        return {
            "tasks": tasks,
            "experiment_id": experiment_id,
            "workspace_dir": str(workspace),
            "code_result": "",
            "experiment_result": {},
            "validation_passed": None,
            "validation_notes": "",
        }

    def code_generator(self, state: CycleState) -> dict:
        artifacts = write_code_artifacts(state, self.repo_root)
        return {"code_result": artifacts["plan_path"], "workspace_dir": artifacts["workspace"]}

    def experiment_runner(self, state: CycleState) -> dict:
        return {"experiment_result": run_experiment(state, self.backend)}

    def science_validator(self, state: CycleState) -> dict:
        passed, notes = validate_science(state)
        return {"validation_passed": passed, "validation_notes": notes}

    def memory_writer(self, state: CycleState) -> dict:
        task = state["tasks"][state.get("task_index", 0)]
        passed = bool(state.get("validation_passed"))
        status = "validated" if passed else "failed"
        self.store.add(
            ExperimentRecord(
                experiment_id=state["experiment_id"],
                subtask=task["subtask"],
                hypothesis=task["hypothesis"],
                result=state.get("validation_notes", ""),
                validation_passed=passed,
                timestamp=datetime.now(timezone.utc).isoformat(),
                validation_criterion=task["validation_criterion"],
                status=status,
                notes=str(state.get("experiment_result")),
                metadata={"attempt": state.get("attempt", 0)},
            )
        )
        last_error = "" if passed else (state.get("validation_notes") or "validation failed")
        return {
            "memory_summary": self.store.summary(),
            "last_error": last_error,
            "cycle_status": status,
        }

    def open_pr(self, state: CycleState) -> dict:
        task = state["tasks"][state.get("task_index", 0)]
        summary = (
            f"## Learned\n\n{task['subtask']}\n\n"
            f"Hypothesis: {task['hypothesis']}\n\n"
            f"Validation: {state.get('validation_notes')}\n\n"
            f"Delta: {state.get('delta')}\n"
        )
        workspace = Path(state.get("workspace_dir") or ".")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "SUMMARY.md").write_text(summary, encoding="utf-8")
        pr_url = propose_pull_request(
            self.repo_root,
            state["experiment_id"],
            f"agent: {task['subtask'][:72]}",
            summary,
            bool(state.get("open_pr")),
        )
        return {"pr_url": pr_url, "summary": summary, "cycle_status": "succeeded"}


def route_after_memory(state: CycleState) -> str:
    passed = bool(state.get("validation_passed"))
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 2))
    task_index = int(state.get("task_index", 0))
    tasks = state.get("tasks") or []
    if not passed:
        if attempt + 1 < max_attempts:
            return "retry"
        return "end"
    if task_index + 1 < len(tasks):
        return "next_task"
    return "pr"


def bump_attempt(state: CycleState) -> dict:
    return {"attempt": int(state.get("attempt", 0)) + 1}


def next_task(state: CycleState) -> dict:
    return {
        "task_index": int(state.get("task_index", 0)) + 1,
        "attempt": 0,
        "last_error": "",
        "experiment_id": uuid4().hex[:12],
    }


def build_graph(runtime: AgentRuntime):
    graph = StateGraph(CycleState)
    graph.add_node("intake", runtime.intake)
    graph.add_node("planner", runtime.planner)
    graph.add_node("code_generator", runtime.code_generator)
    graph.add_node("experiment_runner", runtime.experiment_runner)
    graph.add_node("science_validator", runtime.science_validator)
    graph.add_node("memory_writer", runtime.memory_writer)
    graph.add_node("open_pull_request", runtime.open_pr)
    graph.add_node("bump_attempt", bump_attempt)
    graph.add_node("next_task", next_task)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "code_generator")
    graph.add_edge("code_generator", "experiment_runner")
    graph.add_edge("experiment_runner", "science_validator")
    graph.add_edge("science_validator", "memory_writer")
    graph.add_conditional_edges(
        "memory_writer",
        route_after_memory,
        {
            "retry": "bump_attempt",
            "next_task": "next_task",
            "pr": "open_pull_request",
            "end": END,
        },
    )
    graph.add_edge("bump_attempt", "planner")
    graph.add_edge("next_task", "code_generator")
    graph.add_edge("open_pull_request", END)
    return graph.compile()
