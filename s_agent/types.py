"""
Type definitions for the Astro-Flow-3D agent system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import datetime


class TaskStatus(str, Enum):
    """Task lifecycle status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class TaskRequest:
    """A structured request for a task to be executed by the agent."""
    task_id: str
    parent_task_id: Optional[str] = None
    objective: str
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    created_at: datetime.datetime
    artifact_signature: Optional[str] = None


@dataclass
class TaskResult:
    """The result of a task execution."""
    task_id: str
    status: TaskStatus
    outputs: Dict[str, Any]
    stdout: str
    stderr: str
    exit_code: int
    runtime_seconds: float
    command_hash: Optional[str] = None
    validation_report: Optional["ValidationReport"] = None


@dataclass
class ValidationReport:
    """A structured report from a validator."""
    task_id: str
    status: ValidationStatus
    metrics: Dict[str, Any]
    failures: List[Dict[str, Any]]
    created_at: datetime.datetime


@dataclass
class ArtifactRecord:
    """Metadata about an artifact produced by the agent."""
    artifact_id: str
    task_id: str
    path: str
    hash: str
    size_bytes: int
    created_at: datetime.datetime
    metadata: Dict[str, Any]


@dataclass
class RunManifest:
    """Metadata about a complete run of the agent."""
    run_id: str
    task_id: str
    created_at: datetime.datetime
    git_sha: Optional[str]
    environment_snapshot: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    command: str
    result_path: str
    validation_report_path: str
    artifacts: List[ArtifactRecord]
