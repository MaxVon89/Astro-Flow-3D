"""Layer 1: unstructured research → structured pipeline delta."""

from __future__ import annotations

from pathlib import Path

from .llm import LLMClient, extract_json
from .prompts import INTAKE_SYSTEM
from .state import CycleState, ResearchDelta


def load_project_status(repo_root: str | Path) -> str:
    root = Path(repo_root)
    chunks: list[str] = []
    status_file = root / "Agent" / "state" / "project_status.md"
    readme = root / "README.md"
    if status_file.exists():
        chunks.append(status_file.read_text(encoding="utf-8")[:4000])
    if readme.exists():
        chunks.append(readme.read_text(encoding="utf-8")[:4000])
    return "\n\n".join(chunks) or "(no README or status file)"


def intake_research(llm: LLMClient, state: CycleState) -> ResearchDelta:
    user = (
        f"CURRENT PROJECT STATUS:\n{state.get('project_status', '')}\n\n"
        f"EXPERIMENT MEMORY:\n{state.get('memory_summary', '')}\n\n"
        f"INCOMING RESEARCH:\n{state['research_input']}"
    )
    parsed = extract_json(llm.complete(INTAKE_SYSTEM, user))
    assumptions = parsed.get("assumptions") or []
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    return ResearchDelta(
        method=str(parsed.get("method", "")),
        dataset=str(parsed.get("dataset", "")),
        claimed_improvement=str(parsed.get("claimed_improvement", "")),
        assumptions=[str(item) for item in assumptions],
        pipeline_delta=str(parsed.get("pipeline_delta", "")),
    )
