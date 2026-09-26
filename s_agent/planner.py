from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .schemas import Task


class RuleBasedPlanner:
    """Deterministic fallback planner for the Astro-Flow-3D local agent."""

    def plan(self, objective: str, repo_root: str | Path | None = None) -> List[Task]:
        repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        objective_lower = (objective or "").lower()

        tasks: List[Task] = [
            Task(
                task_id="repo_context",
                objective="Inspect the repository and map the data pipeline.",
                tool="repo_context",
                args={"repo_root": str(repo_root)},
                expected_outputs=["llm_plan.json", "repo_snapshot.json"],
                risk_level="low",
            )
        ]

        if any(token in objective_lower for token in ["repo", "inspect", "summary", "context"]):
            return tasks

        if any(token in objective_lower for token in ["normalize", "preprocess", "science image"]):
            tasks.append(
                Task(
                    task_id="normalize_image",
                    objective="Normalize the JWST science image for downstream tile generation.",
                    tool="normalize",
                    args={
                        "repo_root": str(repo_root),
                        "output_dir": str(repo_root / "artifacts" / "normalize"),
                        "lower_percentile": 1.0,
                        "upper_percentile": 99.8,
                    },
                    expected_outputs=["normalized_image.npy"],
                    risk_level="low",
                )
            )
            return tasks

        if any(token in objective_lower for token in ["tile", "reconstruct", "patch"]):
            tasks.append(
                Task(
                    task_id="generate_tiles",
                    objective="Generate a tile set from the normalized image and validate it.",
                    tool="generate_tiles",
                    args={
                        "repo_root": str(repo_root),
                        "output_dir": str(repo_root / "artifacts" / "tiles"),
                        "tile_size": 256,
                        "stride": 128,
                        "source_name": "synthetic",
                    },
                    expected_outputs=["tile_00000.npy"],
                    risk_level="medium",
                )
            )
            return tasks

        if any(token in objective_lower for token in ["dataset", "build", "manifest"]):
            tasks.append(
                Task(
                    task_id="build_dataset",
                    objective="Build a dataset manifest and validate the generated artifacts.",
                    tool="build_dataset",
                    args={
                        "repo_root": str(repo_root),
                        "input_root": str(repo_root),
                        "output_root": str(repo_root / "artifacts" / "dataset"),
                        "tile_size": 256,
                        "stride": 256,
                    },
                    expected_outputs=["manifest.json", "index.csv"],
                    risk_level="medium",
                )
            )
            return tasks

        tasks.append(
            Task(
                task_id="normalize_image",
                objective="Normalize the image and prepare it for the next stage in the pipeline.",
                tool="normalize",
                args={
                    "repo_root": str(repo_root),
                    "output_dir": str(repo_root / "artifacts" / "normalize"),
                    "lower_percentile": 1.0,
                    "upper_percentile": 99.8,
                },
                expected_outputs=["normalized_image.npy"],
                risk_level="low",
            )
        )
        return tasks
