from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

RiskLevel = Literal["low", "medium", "high"]
RunStatus = Literal[
    "queued",
    "context_loaded",
    "planned",
    "awaiting_approval",
    "approved",
    "running",
    "validated",
    "replanning",
    "failed",
    "completed",
    "halted",
]
ValidationState = Literal["pass", "fail", "error"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskStep:
    task_id: str
    objective: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    risk_level: RiskLevel = "low"
    command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommandRecord:
    command: str
    cwd: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    path: str
    kind: str
    sha256: str
    size_bytes: int
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    task_id: str
    status: ValidationState
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    task_id: str
    created_at: str = field(default_factory=now_iso)
    git_sha: Optional[str] = None
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    command_history: List[CommandRecord] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)
    validation_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "command_history": [c.to_dict() for c in self.command_history],
        }


@dataclass
class AgentState:
    run_id: str
    task_id: str
    objective: str
    status: RunStatus = "queued"
    repo_root: str = ""
    user_request: str = ""
    repo_snapshot: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    plan: List[TaskStep] = field(default_factory=list)
    selected_tools: List[str] = field(default_factory=list)
    approval_status: Literal["pending", "approved", "rejected", "manual_review"] = "pending"
    command_history: List[CommandRecord] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    run_manifest: Optional[RunManifest] = None
    memory_log: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    next_action: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        state = asdict(self)
        state["plan"] = [step.to_dict() for step in self.plan]
        state["command_history"] = [step.to_dict() for step in self.command_history]
        state["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        if self.validation_report is not None:
            state["validation_report"] = self.validation_report.to_dict()
        if self.run_manifest is not None:
            state["run_manifest"] = self.run_manifest.to_dict()
        return state
