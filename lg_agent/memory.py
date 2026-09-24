from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MemoryWriter:
    """Lightweight, file-backed memory layer for the Astro-Flow-3D multi-agent system."""

    def __init__(self, repo_root: str | Path, base_dir: str | Path | None = None):
        self.repo_root = Path(repo_root)
        self.base_dir = Path(base_dir) if base_dir is not None else self.repo_root / ".astroflow_agent"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.base_dir / "run_history.jsonl"
        self.state_path = self.base_dir / "state.json"

    def append_run(self, record: Dict[str, Any]) -> Path:
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return self.run_log_path

    def save_state(self, state: Dict[str, Any]) -> Path:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return self.state_path

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def load_runs(self) -> List[Dict[str, Any]]:
        if not self.run_log_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.run_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
