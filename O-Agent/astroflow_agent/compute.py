"""GPU backends: the agent orchestrates jobs; it is never the GPU process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class JobHandle:
    job_id: str
    backend: str
    status: str
    result: dict[str, Any]


class ComputeBackend(Protocol):
    name: str

    def submit(self, job: dict[str, Any]) -> JobHandle: ...

    def poll(self, handle: JobHandle) -> JobHandle: ...


class LocalHardcodedBackend:
    """First-cycle execution: run local probes instead of submitting A100 jobs."""

    name = "local"

    def submit(self, job: dict[str, Any]) -> JobHandle:
        return JobHandle(
            job_id=str(job.get("experiment_id", "local")),
            backend=self.name,
            status="completed",
            result={"mode": "hardcoded_local", "command": job.get("command", "")},
        )

    def poll(self, handle: JobHandle) -> JobHandle:
        return handle


class RemoteGPUBackend:
    """Modal/RunPod-shaped interface. Submit/poll only after human GPU approval."""

    def __init__(self, provider: str = "modal"):
        self.name = provider

    def submit(self, job: dict[str, Any]) -> JobHandle:
        raise RuntimeError(
            f"{self.name} submission is not wired. Approve GPU and implement "
            "submit/poll against Modal or RunPod; do not train inside the agent process."
        )

    def poll(self, handle: JobHandle) -> JobHandle:
        return handle


def backend_from_name(name: str) -> ComputeBackend:
    if name in {"modal", "runpod"}:
        return RemoteGPUBackend(name)
    return LocalHardcodedBackend()
