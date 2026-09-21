"""Run one research-to-code cycle.

Examples:
    python Agent/run_cycle.py --demo --research "asinh flux-preserving normalization"
    python Agent/run_cycle.py --research "..."   # requires OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from astroflow_agent.compute import backend_from_name
from astroflow_agent.graph import AgentRuntime, build_graph
from astroflow_agent.llm import OpenAICompatibleClient, SequenceLLM
from astroflow_agent.memory import ExperimentStore

DEMO_INTAKE = json.dumps(
    {
        "method": "asinh stretch (Lupton / SDSS)",
        "dataset": "synthetic positive flux arrays standing in for JWST I2D MJy/sr",
        "claimed_improvement": "Preserve inter-band flux ratios that percentile min-max destroys",
        "assumptions": ["Shared stretch parameter across bands", "Linear flux in the unstretched SCI image"],
        "pipeline_delta": "Add a flux-preserving normalization path in src/data/preprocess.py before any second NIRCam filter is stacked as a channel.",
    }
)

DEMO_PLAN = json.dumps(
    {
        "tasks": [
            {
                "subtask": "Quantify flux-ratio preservation of asinh vs percentile clipping",
                "hypothesis": "A shared linear scale keeps band_A/band_B; independent [0,1] clipping does not. Asinh is a display stretch and is not ratio-preserving unless inverted.",
                "validation_criterion": "On synthetic two-band flux, shared linear median relative ratio error is within 1% and independent percentile clipping is worse.",
                "target_files": ["src/data/preprocess.py"],
                "requires_gpu": False,
            }
        ]
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Astro-Flow research-to-code cycle")
    parser.add_argument("--research", required=True, help="Paper abstract, note, or finding")
    parser.add_argument("--state", default="Agent/state/experiments.json")
    parser.add_argument("--demo", action="store_true", help="Use canned intake/plan JSON (no API key)")
    parser.add_argument("--approve-gpu", action="store_true")
    parser.add_argument("--open-pr", action="store_true", help="Call gh pr create (never pushes main)")
    parser.add_argument("--backend", default="local", choices=["local", "modal", "runpod"])
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    if args.demo:
        llm = SequenceLLM([DEMO_INTAKE, DEMO_PLAN])
    else:
        llm = OpenAICompatibleClient()

    runtime = AgentRuntime(
        llm=llm,
        store=ExperimentStore(REPO_ROOT / args.state),
        repo_root=REPO_ROOT,
        backend=backend_from_name(args.backend),
    )
    graph = build_graph(runtime)
    result = graph.invoke(
        {
            "research_input": args.research,
            "gpu_approved": args.approve_gpu,
            "open_pr": args.open_pr,
            "max_attempts": args.max_attempts,
        }
    )
    print(json.dumps(
        {
            "cycle_status": result.get("cycle_status"),
            "experiment_id": result.get("experiment_id"),
            "validation_passed": result.get("validation_passed"),
            "validation_notes": result.get("validation_notes"),
            "pr_url": result.get("pr_url"),
            "tasks": result.get("tasks"),
        },
        indent=2,
    ))
    return 0 if result.get("cycle_status") in {"succeeded", "validated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
