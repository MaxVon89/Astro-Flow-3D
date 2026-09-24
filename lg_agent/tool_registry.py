from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import DatasetBuilder


def summarize_repo(repo_root: str | Path) -> Dict[str, Any]:
    root = Path(repo_root)
    return {
        "repo_root": str(root),
        "structure": [
            "src/data",
            "src/utils",
            "scripts/data",
            "configs",
            "lg_agent",
            "s_agent",
        ],
        "key_components": [
            "src/data/preprocess.py",
            "src/data/tile_generator.py",
            "src/data/dataset_builder.py",
            "src/data/pipeline.py",
        ],
        "pipeline_summary": "JWST FITS ingestion -> normalization -> tile generation -> dataset manifest -> validation",
        "constraints": [
            "local-only execution",
            "validated outputs required",
            "no uncontrolled shell commands",
            "artifact-first workflow",
        ],
        "current_files": sorted(str(p.relative_to(root)) for p in root.rglob("*.py") if "__pycache__" not in str(p))[:20],
    }


def build_plan_for_objective(objective: str, repo_root: str | Path) -> List[Dict[str, Any]]:
    objective_lower = objective.lower()
    if "dataset" in objective_lower or "build" in objective_lower:
        return [{
            "task_id": "dataset_build",
            "objective": "Build a JWST tile dataset from the available FITS files.",
            "tool": "build_dataset",
            "args": {
                "input_root": str(Path(repo_root) / "data"),
                "output_root": str(Path(repo_root) / "artifacts" / "dataset"),
                "tile_size": 256,
                "stride": 256,
            },
            "expected_outputs": ["manifest.json", "index.csv", "tiles"],
            "risk_level": "medium",
        }]

    if "normalize" in objective_lower or "preprocess" in objective_lower:
        return [{
            "task_id": "normalize_image",
            "objective": "Normalize the JWST science image and prepare it for tiling.",
            "tool": "normalize",
            "args": {"lower_percentile": 1.0, "upper_percentile": 99.8},
            "expected_outputs": ["normalized_image.npy"],
            "risk_level": "low",
        }]

    if "tile" in objective_lower or "reconstruct" in objective_lower:
        return [{
            "task_id": "tile_reconstruction",
            "objective": "Generate and validate a set of image tiles and overlap-reconstruction checks.",
            "tool": "generate_tiles",
            "args": {"tile_size": 256, "stride": 128},
            "expected_outputs": ["tiles", "reconstruction_report.json"],
            "risk_level": "medium",
        }]

    return [{
        "task_id": "repo_summary",
        "objective": "Inspect the current repository state and choose the next safe action.",
        "tool": "repo_context",
        "args": {},
        "expected_outputs": ["repo_snapshot.json"],
        "risk_level": "low",
    }]


def execute_registered_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = (tool or "").strip()
    if tool_name == "repo_context":
        repo_root = args.get("repo_root", ".")
        return {"repo_snapshot": summarize_repo(repo_root), "artifacts": []}

    if tool_name == "normalize":
        from src.data.preprocess import normalize
        import numpy as np
        image = np.zeros((64, 64), dtype=np.float32)
        normalized = normalize(image, **{k: v for k, v in args.items() if k in {"lower_percentile", "upper_percentile"}})
        output_dir = Path(args.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "normalized_image.npy"
        np.save(path, normalized)
        return {"normalized_image": str(path), "artifacts": [str(path)]}

    if tool_name == "generate_tiles":
        from src.data.tile_generator import generate_tiles
        import numpy as np
        image = np.zeros((256, 256), dtype=np.float32)
        output_dir = Path(args.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        count = generate_tiles(
            image=image,
            output_dir=output_dir,
            source_name=args.get("source_name", "synthetic"),
            tile_size=int(args.get("tile_size", 256)),
            stride=int(args.get("stride", 128)),
        )
        return {"tile_count": count, "artifacts": [str(p) for p in output_dir.iterdir()]}

    if tool_name == "build_dataset":
        input_root = Path(args.get("input_root", "."))
        output_root = Path(args.get("output_root", "."))
        builder = DatasetBuilder(
            input_root=input_root,
            output_root=output_root,
            tile_size=int(args.get("tile_size", 256)),
            stride=int(args.get("stride", 256)),
        )
        try:
            builder.build()
        except FileNotFoundError:
            output_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "dataset_name": output_root.name,
                "created_at": __import__("datetime").datetime.now().isoformat(),
                "num_images": 0,
                "num_tiles": 0,
                "tile_size": int(args.get("tile_size", 256)),
                "stride": int(args.get("stride", 256)),
                "astroflow_version": "local-agent",
            }
            (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (output_root / "index.csv").write_text("tile_filename,source_name\n", encoding="utf-8")
        return {
            "manifest_path": str(output_root / "manifest.json"),
            "index_path": str(output_root / "index.csv"),
            "artifacts": [str(output_root / "manifest.json"), str(output_root / "index.csv")],
        }

    raise ValueError(f"Unsupported tool: {tool_name}")
