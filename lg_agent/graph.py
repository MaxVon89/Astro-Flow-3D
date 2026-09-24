from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from lg_agent.executor import SafeExecutor
from lg_agent.llm import LocalQwenPlanner
from lg_agent.memory import MemoryWriter
from lg_agent.schemas import AgentState, RunManifest, TaskStep, now_iso
from lg_agent.tool_registry import build_plan_for_objective, summarize_repo
from lg_agent.validator import OutputValidator

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    StateGraph = None
    END = "__end__"


class AstroFlowMultiAgent:
    """LangGraph-based multi-agent orchestrator for Astro-Flow-3D."""

    def __init__(self, repo_root: str | Path, dry_run: bool = False):
        self.repo_root = Path(repo_root)
        self.executor = SafeExecutor(self.repo_root, dry_run=dry_run)
        self.memory = MemoryWriter(self.repo_root)
        self.validator = OutputValidator(self.repo_root)
        self.llm_planner = LocalQwenPlanner(model_name="qwen-code-3.5-30b")
        self.state = self._new_state()

    def _new_state(self) -> AgentState:
        stamp = now_iso()
        return AgentState(
            run_id=f"run_{hashlib.sha256(stamp.encode()).hexdigest()[:12]}",
            task_id=f"task_{hashlib.sha256(stamp.encode()).hexdigest()[:12]}",
            objective="",
            repo_root=str(self.repo_root),
            user_request="",
            status="queued",
            max_retries=3,
        )

    @staticmethod
    def _to_task_step(step: Any) -> TaskStep:
        if isinstance(step, TaskStep):
            return step
        if isinstance(step, dict):
            return TaskStep(**step)
        raise TypeError(f"Unsupported step type: {type(step)!r}")

    @staticmethod
    def _to_serializable_state(state: Dict[str, Any]) -> Dict[str, Any]:
        serializable = dict(state)
        if "plan" in serializable:
            serializable["plan"] = [
                step.to_dict() if hasattr(step, "to_dict") else step for step in serializable["plan"]
            ]
        if "command_history" in serializable:
            serializable["command_history"] = [
                item.to_dict() if hasattr(item, "to_dict") else item for item in serializable["command_history"]
            ]
        if "artifacts" in serializable:
            serializable["artifacts"] = [
                item.to_dict() if hasattr(item, "to_dict") else item for item in serializable["artifacts"]
            ]
        if "validation_report" in serializable and serializable["validation_report"] is not None:
            serializable["validation_report"] = serializable["validation_report"].to_dict()
        if "run_manifest" in serializable and serializable["run_manifest"] is not None:
            serializable["run_manifest"] = serializable["run_manifest"].to_dict()
        return serializable

    def node_repo_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["repo_snapshot"] = summarize_repo(self.repo_root)
        state["status"] = "context_loaded"
        return state

    def node_planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        llm_steps = self.llm_planner.plan(state.get("objective", ""), state.get("repo_snapshot", {}))
        if llm_steps:
            steps = llm_steps
        else:
            steps = build_plan_for_objective(state.get("objective", ""), self.repo_root)
        state["plan"] = [self._to_task_step(step) for step in steps]
        state["selected_tools"] = [step.tool for step in state["plan"]]
        state["status"] = "planned"
        return state

    def node_approval_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        for step in state.get("plan", []):
            normalized = self._to_task_step(step)
            if normalized.risk_level == "high":
                state["approval_status"] = "manual_review"
                state["status"] = "awaiting_approval"
                return state
        state["approval_status"] = "approved"
        state["status"] = "approved"
        return state

    def node_executor(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["status"] = "running"
        for raw_step in state.get("plan", []):
            step = self._to_task_step(raw_step)
            try:
                result = self.executor.execute_step(step)
            except Exception as exc:  # pragma: no cover
                state["error"] = str(exc)
                state["status"] = "failed"
                return state
            state.setdefault("artifact_paths", [])
            state["artifact_paths"].extend(result.get("result", {}).get("artifacts", []))
            state.setdefault("command_history", [])
            state["command_history"].extend(self.executor.command_history)
        state["artifact_paths"] = sorted(set(state.get("artifact_paths", [])))
        return state

    def node_validator(self, state: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {"artifacts": state.get("artifact_paths", [])}
        planned_steps = [self._to_task_step(step) for step in state.get("plan", [])]
        expected_outputs = [item for step in planned_steps for item in step.expected_outputs]
        report = self.validator.validate(state.get("task_id", "unknown"), outputs, expected_outputs)
        state["validation_report"] = report
        state["status"] = "validated"
        return state

    def node_memory_writer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        manifest = RunManifest(
            run_id=state.get("run_id", "unknown"),
            task_id=state.get("task_id", "unknown"),
            created_at=now_iso(),
            git_sha=self._git_sha(),
            environment_snapshot={"python": sys_version(), "repo_root": str(self.repo_root)},
            config_snapshot={"dry_run": self.executor.dry_run, "max_retries": state.get("max_retries", 3)},
            command_history=state.get("command_history", []),
            artifact_paths=state.get("artifact_paths", []),
            validation_report_path=None,
        )
        state["run_manifest"] = manifest
        self.memory.append_run({
            "run_id": state.get("run_id"),
            "task_id": state.get("task_id"),
            "status": state.get("status"),
            "objective": state.get("objective"),
            "artifacts": state.get("artifact_paths", []),
            "validation_report": None if state.get("validation_report") is None else state["validation_report"].to_dict(),
            "timestamp": now_iso(),
        })
        self.memory.save_state(self._to_serializable_state(state))
        state["status"] = "completed" if state.get("validation_report") and state["validation_report"].status == "pass" else "failed"
        return state

    def node_replanner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        state["status"] = "replanning"
        state["next_action"] = "retry_safe_plan"
        if retry_count >= state.get("max_retries", 3):
            state["status"] = "halted"
            state["error"] = "Max retries reached."
        return state

    def _git_sha(self) -> Optional[str]:
        try:
            completed = subprocess.run(["git", "-C", str(self.repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            if completed.returncode == 0:
                return completed.stdout.strip()
        except Exception:
            pass
        return None

    def run_task(self, objective: str, user_request: Optional[str] = None) -> Dict[str, Any]:
        state = self._new_state().to_dict()
        state["objective"] = objective
        state["user_request"] = user_request or objective
        if StateGraph is not None:
            app = self.build_langgraph_app()
            return app.invoke(state)
        state = self.node_repo_context(state)
        state = self.node_planner(state)
        state = self.node_approval_gate(state)
        if state.get("status") == "awaiting_approval":
            return state
        state = self.node_executor(state)
        if state.get("status") == "failed":
            state = self.node_replanner(state)
            return state
        state = self.node_validator(state)
        if state.get("validation_report") and state["validation_report"].status != "pass":
            state = self.node_replanner(state)
            return state
        state = self.node_memory_writer(state)
        return state

    def build_langgraph_app(self):
        if StateGraph is None:
            return None

        builder = StateGraph(dict)
        builder.add_node("repo_context", self.node_repo_context)
        builder.add_node("planner", self.node_planner)
        builder.add_node("approval_gate", self.node_approval_gate)
        builder.add_node("executor", self.node_executor)
        builder.add_node("validator", self.node_validator)
        builder.add_node("memory_writer", self.node_memory_writer)
        builder.add_node("replanner", self.node_replanner)
        builder.add_node("halt", lambda state: {**state, "status": "halted"})

        builder.set_entry_point("repo_context")
        builder.add_edge("repo_context", "planner")
        builder.add_edge("planner", "approval_gate")
        builder.add_conditional_edges(
            "approval_gate",
            lambda st: "executor" if st.get("approval_status") == "approved" else "halt",
            {"executor": "executor", "halt": "halt"},
        )
        builder.add_edge("executor", "validator")
        builder.add_conditional_edges(
            "validator",
            lambda st: "memory_writer" if st.get("validation_report") and st["validation_report"].status == "pass" else "replanner",
            {"memory_writer": "memory_writer", "replanner": "replanner"},
        )
        builder.add_conditional_edges(
            "replanner",
            lambda st: "planner" if st.get("retry_count", 0) < st.get("max_retries", 3) else "halt",
            {"planner": "planner", "halt": "halt"},
        )
        builder.add_edge("memory_writer", END)
        builder.add_edge("halt", END)
        return builder.compile()

    def build_langgraph_definition(self) -> Dict[str, Any]:
        return {
            "graph": ["repo_context", "planner", "approval_gate", "executor", "validator", "memory_writer", "replanner", "halt"],
            "edges": [
                ("repo_context", "planner"),
                ("planner", "approval_gate"),
                ("approval_gate", "executor"),
                ("executor", "validator"),
                ("validator", "memory_writer"),
                ("validator", "replanner"),
                ("replanner", "planner"),
            ],
        }


def sys_version() -> str:
    try:
        import platform
        return platform.python_version()
    except Exception:
        return "unknown"
