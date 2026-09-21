# Astro-Flow Agent

Research-to-code loop for Astro-Flow-3D. Incoming papers and notes become a pipeline delta, a science-aware plan, local or GPU-orchestrated experiments, physical validation, experiment memory, and a review PR. The graph never pushes `main`.

```text
research → intake → planner → code workspace → experiment → science validator
                ↑                    │
                └── retry on fail ───┘
                         │ pass
                         ▼
                   memory + PR summary
```

## Stack

- **LangGraph** for typed nodes and the validation-retry conditional edge
- **JSON memory** at `Agent/state/experiments.json` (`experiment_id`, `subtask`, `hypothesis`, `result`, `validation_passed`, `timestamp`)
- **Local hardcoded execution** for the first cycles (flux-ratio probe against `src/data/preprocess.py`)
- **Modal / RunPod** as submit/poll interfaces only — the agent does not become the GPU process
- **`gh pr create`** when `--open-pr` is set; default is a dry-run branch name

## Install

```powershell
pip install -r Agent/requirements.txt
```

## Run

Canned intake/plan (no API key), real scientific probe:

```powershell
python Agent/run_cycle.py --demo --research "Test flux-preserving asinh normalization"
```

Model-backed intake and planner:

```powershell
$env:OPENAI_API_KEY = "..."
python Agent/run_cycle.py --research "Paste an abstract or note"
```

GPU jobs stay blocked until `--approve-gpu`. `--backend modal` or `--backend runpod` will refuse until those submit/poll clients are implemented.

## Tests

```powershell
python -m unittest Agent.tests.test_agent
```

From repo root you can also run:

```powershell
python Agent/tests/test_agent.py
```
