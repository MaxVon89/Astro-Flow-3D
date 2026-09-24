I’m turning the repo plan into a concrete four-day agent workflow you can hand to the next agent, with explicit commands and the exact deliverables for each day.

## Four-day execution plan for the agent

The next four days should focus on building a small, safe local agent loop that works with the existing repo pipeline in preprocess.py, tile_generator.py, dataset_builder.py, and pipeline.py, not on a full autonomous architecture.

The goal is to get to a working local “planner → executor → validator → memory” loop that can:
- inspect the repo,
- create structured task plans,
- run only approved local commands,
- validate produced artifacts,
- persist provenance.

This matches the intent in Agent-Plan.md.

---

## Day 1 — Define the agent contract and repo-aware JSON schema

### Tasks
1. Create the package skeleton for the local agent.
2. Define a minimal task schema:
   - task_id
   - objective
   - tool
   - args
   - expected_outputs
   - risk_level
3. Define a result schema:
   - status
   - artifacts
   - metrics
   - validation
   - provenance
4. Define a validator report schema.
5. Build a small repo inspection function that reads the current pipeline structure and returns a repo summary.

### Deliverable
- A working local package with schemas and a repo-aware planner stub.

### Agent instructions for Day 1
“Read Agent-Plan.md, preprocess.py, tile_generator.py, dataset_builder.py, and pipeline.py. Create the local agent package with typed JSON schemas for task, result, validation report, and run provenance. Implement a repo inspection function that summarizes the project and exposes the current preprocessing/tile/dataset pipeline. Do not add arbitrary shell execution. Use only local Python logic and keep the interface deterministic.”

### Commands
```bash
cd /home/nvidia/mmk/Astro-Flow-3D
mkdir -p s_agent/examples
find src -maxdepth 3 -type f | sort
python - <<'PY'
import json, pathlib
root = pathlib.Path(".")
for p in sorted(root.glob("src/**/*.py")):
    print(p)
PY
```

---

## Day 2 — Safe executor + memory + provenance

### Tasks
1. Implement a local safe executor:
   - allowed commands only
   - stdout/stderr capture
   - exit codes
   - dry-run mode
2. Implement a memory store:
   - task history
   - annotation of artifacts
   - git sha if available
   - timestamp and environment snapshot
3. Persist provenance for every task run.
4. Add a rule-based planner fallback for when the local LLM is unavailable.

### Deliverable
- The agent can run a harmless local task and persist the result.

### Agent instructions for Day 2
“Build the safe local executor and memory layer. Keep it repo-aware and deterministic. The executor must reject dangerous commands, capture structured results, and allow dry-run mode. Memory must store task state plus provenance fields: task_id, run_id, timestamp, git_sha, environment snapshot, and artifact references. Add a fallback plan generator that does not require the local LLM.”

### Commands
```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python - <<'PY'
import pathlib, json
root = pathlib.Path("s_agent")
root.mkdir(exist_ok=True)
for name in ["__init__.py","schemas.py","memory.py","executor.py","planner.py","agent.py"]:
    p = root / name
    if not p.exists():
        p.write_text("# placeholder\\n", encoding="utf-8")
print("agent scaffold ready")
PY
```

```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python -m pytest -q
```

---

## Day 3 — Integrate with the repo pipeline

### Tasks
1. Implement a planner that maps repo tasks to pipeline functions:
   - preprocessing
   - tile generation
   - dataset build
   - validation
2. Add a local model adapter for qwen-code-3.5-30b as the optional planner backend.
3. Keep deterministic rule-based planner as the primary fallback.
4. Wire the planner to repo functions instead of generic placeholder tasks.

### Deliverable
- The agent can create a real plan for:
  - preprocess dataset
  - build tiles
  - validate results

### Agent instructions for Day 3
“Connect the planner to the actual Astro-Flow-3D pipeline. Map one objective to the repo’s existing modules: normalize using the logic in preprocess.py, generate tiles via tile_generator.py, and build a manifest via dataset_builder.py. Keep task generation JSON-typed and deterministic. If the local model fails, fall back to the rule-based planner.”

### Commands
```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python - <<'PY'
from pathlib import Path
for p in ["s_agent/schemas.py","s_agent/planner.py","s_agent/executor.py","s_agent/memory.py","s_agent/agent.py"]:
    if not Path(p).exists():
        raise SystemExit(f"Missing {p}")
print("core files exist")
PY
```

```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python - <<'PY'
from src.data.preprocess import normalize
import numpy as np
img = np.random.rand(32, 32).astype(np.float32)
out = normalize(img)
print(out.shape, out.min(), out.max())
PY
```

---

## Day 4 — Validation loop and end-to-end demo

### Tasks
1. Add validation checks:
   - expected files exist
   - manifest files are readable
   - tile count matches expectations
   - output metadata is consistent
2. Run a tiny end-to-end sample through the agent loop.
3. Fix failures and write a runbook.
4. Add an example task and output JSON to the local agent package.

### Deliverable
- A working local agent demo that can process a tiny synthetic dataset and produce structured validation output.

### Agent instructions for Day 4
“Build the orchestration loop: Planner → Executor → Validator → Memory. Run one real local task using a tiny synthetic dataset or a small repo fixture. Validate outputs and mark success only when the validation report passes. Produce the run report and example task files. Keep all actions local and safe.”

### Commands
```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python - <<'PY'
import json
from pathlib import Path
task = {
    "task_id": "demo_task_001",
    "objective": "Normalize a synthetic image and validate output",
    "tool": "normalize",
    "args": {"lower_percentile": 1.0, "upper_percentile": 99.8},
    "expected_outputs": ["normalized_image.npy"],
    "risk_level": "low",
}
Path("s_agent/examples").mkdir(exist_ok=True)
Path("s_agent/examples/task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
print("demo task written")
PY
```

```bash
cd /home/nvidia/mmk/Astro-Flow-3D
python -m pytest -q
```

---

## Recommended agent prompt to paste each day

### Day 1 prompt
“Review Agent-Plan.md and the current repo pipeline in preprocess.py, tile_generator.py, dataset_builder.py, and pipeline.py. Build the minimal local agent package in the new local agent directory. Create typed JSON schemas for task, result, validation report, and provenance. Implement a repo-inspection function that summarizes the project and document the current pipeline. Keep everything local, deterministic, and safe. Do not use unrestricted shell execution.”

### Day 2 prompt
“Implement the safe local executor and memory layer. Allow only approved Python-based commands, record stdout/stderr and exit codes, and support dry-run mode. Persist task history and provenance to disk. Add a fallback deterministic planner if the local model is unavailable. Keep execution strictly local.”

### Day 3 prompt
“Connect the planner to the repo’s real alignment functions: preprocessing, tiling, and dataset build. Use the actual project modules as the tool backend. Keep the planner JSON-structured and make the fallback planner deterministic. Do not invent new data pipeline components unless they are needed for compatibility.”

### Day 4 prompt
“Complete the orchestration loop: planner → executor → validator → memory. Run one small end-to-end task using a synthetic or minimal fixture. Validate all outputs before marking success, persist provenance, and write a runbook plus example JSON task/result artifacts.”

---

## Model guidance

For this sprint, use the local qwen model only as the planner/code assistant. It is good for:
- planning tasks
- repo summaries
- task decomposition
- code suggestion

It is not a replacement for:
- scientific validation logic
- image-model encoders
- PSF / deconvolution logic
- heavy vision backbones

So the correct local setup is:
- qwen-code-3.5-30b as planner
- deterministic rule-based planner as fallback
- repo validators as the actual correctness gate
- safe local executor as execution boundary

---

## If you want the shortest possible command set

Use this as the daily agent handoff:

```bash
cd /home/nvidia/mmk/Astro-Flow-3D
mkdir -p s_agent/examples
find src -maxdepth 3 -type f | sort
python -m pytest -q
```

Then give the agent this daily instruction:

“Read Agent-Plan.md, the repo pipeline files, and build the local agent in the new agent package with a planner, executor, validator, and memory layer. Keep every run artifact-driven, local-only, and validated. Do not allow arbitrary shell commands.”

If you want, I can turn this into a single copy-paste “agent brief” that is even shorter and more prescriptive for the next 4 days.

I’m narrowing this to a realistic 8×A100 plan that matches the repo and the six-day sprint, instead of suggesting a generic “big model” project.

## Best use of 8×A100s for this repo

The best project for that compute is not “a bigger local LLM.” It is a distributed multi-band JWST pretraining and reconstruction pipeline built on the existing data flow in preprocess.py, tile_generator.py, dataset_builder.py, and pipeline.py.

This is the kind of workload that actually benefits from 8×A100:
- large tile batches
- multi-band imagery
- self-supervised pretraining
- PSF-aware reconstruction
- validation across many tiles in parallel

---

## The project to build

### 1) Multi-band tile pretraining
Use a masked autoencoder or masked patch model on JWST tiles.

Why it fits:
- You already have tile generation and dataset assembly hooks
- Tile-level masked reconstruction is naturally parallel across GPUs
- It is a meaningful “science + compute” project, not just a toy demo

Model stack:
- ViT-S or ViT-B backbone
- channel adapter for N-band inputs
- MAE decoder
- distributed training with DDP/FSDP

### 2) PSF-aware reconstruction and validation
This is the scientifically valuable part.

Use the A100 cluster to:
- build tiles with overlap
- reconstruct full images with weighted overlap-add
- validate flux conservation and PSF consistency
- compare reconstruction error against tile overlap and band alignment

This is much more useful than generic model benchmarking.

### 3) Multi-band contrastive or masked-band learning
Use bands as masked tokens or channels:
- randomly mask entire bands or patches
- predict missing band information
- train a model that learns cross-band relationships

This is a strong use of 8 GPUs because:
- each batch can contain many tiles and bands
- loss is more complex than simple classification
- validation can be run over dozens of patches per step

---

## Why this is better than a large LLM

The local qwen model should stay as a planner, not as the compute driver.

Use the 8×A100s for:
- vision model training
- self-supervised masking
- tile reconstruction
- metadata-heavy dataset optimization

Do not burn the cluster on:
- a 30B chat model
- broad general-purpose “agentic” coding loops
- unnecessary cloud-style orchestration

The repo’s real scientific value is in the dataset + image modeling path, not in a huge chat model.

---

## Concrete architecture for the next sprint

### Core training pipeline
1. Raw FITS → normalized tiles
2. Tile generation with overlap
3. Band-aware mask / augmentation
4. Distributed MAE pretraining
5. Fine-tune on reconstruction or downstream quality label
6. Validate on science metrics:
   - flux conservation
   - PSF width
   - reconstruction RMSE
   - band-ratio invariance

### Distributed training setup
- DDP or FSDP over 8 GPUs
- per-GPU batch size tuned to tile size
- gradient accumulation for large effective batch
- mixed precision
- dataset sharding across workers

### Good target problem
A project like:
- “Multi-band JWST tile MAE with PSF-aware reconstruction and validation”

This is exactly the kind of workload that justifies a 8×A100 setup.

---

## Best 4-day version of the same idea

If you want something that can be finished in a sprint:

### Day 1
- Finalize the dataset schema and artifact contract
- Create a data loader for tile batches

### Day 2
- Implement multi-band preprocessing and tile generation
- Add overlap and weighted reconstruction path

### Day 3
- Train a lightweight ViT/MAE on synthetic or small real tile data
- Launch distributed training on 8 GPUs

### Day 4
- Validate reconstruction, flux conservation, and band consistency
- Write demo results and metrics

That is a real “big compute” project that produces a meaningful artifact.

---

## What model family to use

For this repo, the right model stack is:
- planner: local qwen-code-3.5-30b
- vision backbone: ViT-S / ViT-B / Swin-T or ResNet50
- pretraining: MAE-style masked reconstruction
- optional head: PSF-aware reconstruction or masked-band prediction

Not an LLM-only stack.

---

## Recommendation

Use the 8×A100s for a distributed multi-band JWST MAE + reconstruction pipeline, not for a giant local agent.

That gives you:
- real GPU utilization
- a meaningful scientific objective
- tight connection to the repo
- outputs that are worth validating and presenting

If you want, I can draft the exact 4-day implementation spec for this 8×A100 project, including the training script structure and the concrete commands to launch the distributed job.