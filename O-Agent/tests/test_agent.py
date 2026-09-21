import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from astroflow_agent.compute import RemoteGPUBackend
from astroflow_agent.graph import AgentRuntime, build_graph, route_after_memory
from astroflow_agent.llm import SequenceLLM
from astroflow_agent.memory import ExperimentRecord, ExperimentStore
from astroflow_agent.science import flux_ratio_report

INTAKE = json.dumps(
    {
        "method": "asinh stretch",
        "dataset": "synthetic flux",
        "claimed_improvement": "preserve ratios",
        "assumptions": ["shared stretch"],
        "pipeline_delta": "change preprocess.py normalization",
    }
)
PLAN_FLUX = json.dumps(
    {
        "tasks": [
            {
                "subtask": "Preserve flux ratios with asinh",
                "hypothesis": "Shared asinh keeps band ratios",
                "validation_criterion": "asinh flux ratio error within 1%",
                "target_files": ["src/data/preprocess.py"],
                "requires_gpu": False,
            }
        ]
    }
)
PLAN_GPU = json.dumps(
    {
        "tasks": [
            {
                "subtask": "Train a two-band masked autoencoder",
                "hypothesis": "Band dropout learns SED structure",
                "validation_criterion": "held-out band beats mean predictor",
                "target_files": ["src/models"],
                "requires_gpu": True,
            }
        ]
    }
)


class AgentArchitectureTests(unittest.TestCase):
    def test_memory_schema_round_trip(self):
        with TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory) / "experiments.json")
            record = store.add(ExperimentRecord.create("task", "h", "criterion"))
            updated = store.update(record.experiment_id, validation_passed=True, status="validated")
            payload = json.loads(Path(directory, "experiments.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["subtask"], "task")
            self.assertTrue(updated.validation_passed)

    def test_percentile_clipping_destroys_flux_ratios(self):
        report = flux_ratio_report()
        self.assertTrue(report["percentile_destroys_ratios"])
        self.assertTrue(report["linear_within_tolerance"])

    def test_graph_validates_flux_and_dry_runs_pr(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("single-band tiling pipeline\n", encoding="utf-8")
            (root / "src" / "data").mkdir(parents=True)
            (root / "src" / "data" / "preprocess.py").write_text("# preprocess\n", encoding="utf-8")
            store = ExperimentStore(root / "experiments.json")
            runtime = AgentRuntime(SequenceLLM([INTAKE, PLAN_FLUX]), store, root)
            result = build_graph(runtime).invoke(
                {"research_input": "asinh flux ratios", "max_attempts": 2, "open_pr": False}
            )
            self.assertTrue(result["validation_passed"])
            self.assertEqual(result["cycle_status"], "succeeded")
            self.assertTrue(str(result["pr_url"]).startswith("dry-run:"))
            self.assertTrue((root / "Agent" / "workspace" / result["experiment_id"] / "SUMMARY.md").exists())

    def test_gpu_requires_human_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("status\n", encoding="utf-8")
            store = ExperimentStore(root / "experiments.json")
            runtime = AgentRuntime(SequenceLLM([INTAKE, PLAN_GPU]), store, root)
            result = build_graph(runtime).invoke(
                {
                    "research_input": "masked band MAE",
                    "gpu_approved": False,
                    "max_attempts": 1,
                }
            )
            self.assertFalse(result["validation_passed"])
            self.assertIn("approve-gpu", result["validation_notes"])

    def test_remote_backend_does_not_run_inside_agent(self):
        backend = RemoteGPUBackend("modal")
        with self.assertRaises(RuntimeError):
            backend.submit({"experiment_id": "x"})

    def test_retry_routing(self):
        self.assertEqual(
            route_after_memory({"validation_passed": False, "attempt": 0, "max_attempts": 2, "tasks": [{}]}),
            "retry",
        )
        self.assertEqual(
            route_after_memory({"validation_passed": True, "task_index": 0, "tasks": [{}, {}]}),
            "next_task",
        )


if __name__ == "__main__":
    unittest.main()
