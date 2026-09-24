import json
from pathlib import Path

from lg_agent.tool_registry import execute_registered_tool


def test_repo_context_writes_expected_artifacts(tmp_path):
    result = execute_registered_tool("repo_context", {"repo_root": str(tmp_path)})

    assert "artifacts" in result
    assert str(tmp_path / "llm_plan.json") in result["artifacts"]
    assert str(tmp_path / "repo_snapshot.json") in result["artifacts"]

    llm_plan = json.loads((tmp_path / "llm_plan.json").read_text(encoding="utf-8"))
    assert llm_plan["repo_root"] == str(tmp_path.resolve())
    assert llm_plan["repo_snapshot"]["repo_root"] == str(tmp_path.resolve())
