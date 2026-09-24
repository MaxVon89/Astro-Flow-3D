"""
JSON schemas for the Astro-Flow-3D agent system.
"""

import json
from typing import Dict, Any
from datetime import datetime


def validate_task_request(data: Dict[str, Any]) -> bool:
    """Validate a TaskRequest schema."""
    required_fields = ['task_id', 'objective', 'inputs', 'expected_outputs', 'created_at']
    for field in required_fields:
        if field not in data:
            return False
    return True


def validate_task_result(data: Dict[str, Any]) -> bool:
    """Validate a TaskResult schema."""
    required_fields = ['task_id', 'status', 'outputs', 'stdout', 'stderr', 'exit_code', 'runtime_seconds']
    for field in required_fields:
        if field not in data:
            return False
    return True


def validate_validation_report(data: Dict[str, Any]) -> bool:
    """Validate a ValidationReport schema."""
    required_fields = ['task_id', 'status', 'metrics', 'failures', 'created_at']
    for field in required_fields:
        if field not in data:
            return False
    return True


def validate_artifact_record(data: Dict[str, Any]) -> bool:
    """Validate an ArtifactRecord schema."""
    required_fields = ['artifact_id', 'task_id', 'path', 'hash', 'size_bytes', 'created_at', 'metadata']
    for field in required_fields:
        if field not in data:
            return False
    return True


def validate_run_manifest(data: Dict[str, Any]) -> bool:
    """Validate a RunManifest schema."""
    required_fields = ['run_id', 'task_id', 'created_at', 'environment_snapshot', 'config_snapshot', 'command', 'result_path', 'validation_report_path', 'artifacts']
    for field in required_fields:
        if field not in data:
            return False
    return True


def task_request_to_json(task_request) -> str:
    """Convert a TaskRequest to JSON."""
    return json.dumps({
        "task_id": task_request.task_id,
        "parent_task_id": getattr(task_request, 'parent_task_id', None),
        "objective": task_request.objective,
        "inputs": task_request.inputs,
        "expected_outputs": task_request.expected_outputs,
        "created_at": task_request.created_at.isoformat(),
        "artifact_signature": getattr(task_request, 'artifact_signature', None)
    })


def task_result_to_json(task_result) -> str:
    """Convert a TaskResult to JSON."""
    return json.dumps({
        "task_id": task_result.task_id,
        "status": task_result.status.value,
        "outputs": task_result.outputs,
        "stdout": task_result.stdout,
        "stderr": task_result.stderr,
        "exit_code": task_result.exit_code,
        "runtime_seconds": task_result.runtime_seconds,
        "command_hash": getattr(task_result, 'command_hash', None),
        "validation_report": task_result.validation_report.to_dict() if task_result.validation_report else None
    })


def validation_report_to_json(validation_report) -> str:
    """Convert a ValidationReport to JSON."""
    return json.dumps({
        "task_id": validation_report.task_id,
        "status": validation_report.status.value,
        "metrics": validation_report.metrics,
        "failures": validation_report.failures,
        "created_at": validation_report.created_at.isoformat()
    })


def artifact_record_to_json(artifact_record) -> str:
    """Convert an ArtifactRecord to JSON."""
    return json.dumps({
        "artifact_id": artifact_record.artifact_id,
        "task_id": artifact_record.task_id,
        "path": artifact_record.path,
        "hash": artifact_record.hash,
        "size_bytes": artifact_record.size_bytes,
        "created_at": artifact_record.created_at.isoformat(),
        "metadata": artifact_record.metadata
    })


def run_manifest_to_json(run_manifest) -> str:
    """Convert a RunManifest to JSON."""
    return json.dumps({
        "run_id": run_manifest.run_id,
        "task_id": run_manifest.task_id,
        "created_at": run_manifest.created_at.isoformat(),
        "git_sha": run_manifest.git_sha,
        "environment_snapshot": run_manifest.environment_snapshot,
        "config_snapshot": run_manifest.config_snapshot,
        "command": run_manifest.command,
        "result_path": run_manifest.result_path,
        "validation_report_path": run_manifest.validation_report_path,
        "artifacts": [artifact_record_to_json(art) for art in run_manifest.artifacts]
    })
