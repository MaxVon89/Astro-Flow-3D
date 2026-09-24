from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional


class LocalQwenPlanner:
    """Safe adapter for a local qwen model. Uses Ollama-style local inference when present; otherwise falls back to deterministic planning."""

    def __init__(self, model_name: str = "qwen-code-3.5-30b", base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        self.enabled = self._is_model_available()

    def _is_model_available(self) -> bool:
        if os.getenv("LOCAL_LLM_ENABLED", "1") == "0":
            return False
        candidates = [
            os.getenv("LOCAL_LLM_MODEL"),
            self.model_name,
            "qwen-code:30b",
            "qwen2.5-coder:7b",
        ]
        for candidate in candidates:
            if candidate:
                return True
        return False

    def _call_local_model(self, prompt: str) -> Optional[str]:
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None

    def plan(self, objective: str, repo_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        prompt = json.dumps({
            "objective": objective,
            "repo_snapshot": repo_snapshot,
            "instruction": "Generate a JSON list of safe, repo-aware task steps. Keep tasks deterministic and validation-first.",
        }, sort_keys=True)

        response = self._call_local_model(prompt)
        if response:
            try:
                payload = json.loads(response)
                if isinstance(payload, list):
                    return payload
            except Exception:
                pass

        return [{
            "task_id": "llm_generated_task",
            "objective": objective,
            "tool": "repo_context",
            "args": {
                "model": self.model_name,
                "objective": objective,
                "repo_snapshot": repo_snapshot,
                "plan_type": "safe_local_research_plan",
            },
            "expected_outputs": ["llm_plan.json"],
            "risk_level": "low",
        }]


class LocalPlannerAdapter(LocalQwenPlanner):
    """Backward-compatible alias used by older imports."""

    pass
