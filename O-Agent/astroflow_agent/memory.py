"""JSON experiment memory that lives in the repo.

Schema follows the architecture note:
experiment_id, subtask, hypothesis, result, validation_passed, timestamp.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ExperimentRecord:
    experiment_id: str
    subtask: str
    hypothesis: str
    result: str
    validation_passed: bool | None
    timestamp: str
    validation_criterion: str = ""
    status: str = "planned"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        subtask: str,
        hypothesis: str,
        validation_criterion: str = "",
        **metadata: Any,
    ) -> ExperimentRecord:
        return cls(
            experiment_id=uuid4().hex[:12],
            subtask=subtask,
            hypothesis=hypothesis,
            result="",
            validation_passed=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            validation_criterion=validation_criterion,
            metadata=metadata,
        )


class ExperimentStore:
    """Append-only JSON list of experiments."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def list(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Experiment memory must be a JSON list: {self.path}")
        return [ExperimentRecord(**item) for item in payload]

    def summary(self, limit: int = 12) -> str:
        records = self.list()[-limit:]
        if not records:
            return "(no prior experiments)"
        lines = []
        for record in records:
            lines.append(
                f"- {record.experiment_id} [{record.status}] "
                f"passed={record.validation_passed} {record.subtask}: {record.result}"
            )
        return "\n".join(lines)

    def _write(self, records: list[ExperimentRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in records], handle, indent=2)
            handle.write("\n")
        temporary.replace(self.path)

    def add(self, record: ExperimentRecord) -> ExperimentRecord:
        records = self.list()
        records.append(record)
        self._write(records)
        return record

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord:
        records = self.list()
        for index, record in enumerate(records):
            if record.experiment_id == experiment_id:
                updated = ExperimentRecord(**{**asdict(record), **changes})
                records[index] = updated
                self._write(records)
                return updated
        raise KeyError(f"Unknown experiment: {experiment_id}")
