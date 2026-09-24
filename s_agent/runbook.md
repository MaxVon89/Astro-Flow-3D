# Astro-Flow-3D Agent Runbook

This document provides instructions for using the local research agent that operates on the Astro-Flow-3D pipeline.

## Overview

The Astro-Flow-3D agent is a local, artifact-driven research system designed to:
- Operate safely without external dependencies
- Execute structured tasks on JWST data processing pipeline
- Validate outputs with scientific and repository invariants
- Persist task history and provenance information

## Installation

The agent requires Python 3.8+ and the dependencies listed in `requirements.txt`.

```bash
# Install dependencies
pip install -r requirements.txt

# The agent is self-contained in the s_agent directory
```

## Usage

### Basic Agent Initialization

```python
from s_agent.agent import create_agent

# Create an agent instance
agent = create_agent(dry_run=False)  # Set to True for dry-run mode
```

### Running a Task

```python
from s_agent.types import TaskRequest
from datetime import datetime

# Define a task
task = TaskRequest(
    task_id="dataset_build_1",
    objective="Build JWST dataset from FITS file",
    inputs={"input_fits": "/path/to/jwst.fits"},
    expected_outputs=["output/index.csv", "output/manifest.json"],
    created_at=datetime.now()
)

# Execute the task
result = agent.process_task(task)
print(f"Task completed with status: {result.status}")
```

### Running Multiple Tasks

```python
# Create multiple tasks
tasks = [
    TaskRequest(...),
    TaskRequest(...),
]

# Run all tasks in sequence
results = agent.run_pipeline(tasks)
```

## Safety Features

1. **Command Allowlist**: Only commands matching the allowlist can be executed
2. **Dry-run Mode**: Set `dry_run=True` to preview what would happen without executing
3. **Validation Required**: All outputs must pass validation before being marked successful
4. **Memory Tracking**: Every task and artifact is stored with provenance information

## Memory System

The agent uses SQLite for persistent storage of:
- Task requests and results
- Artifact metadata
- Run manifests
- Validation reports

Storage location: `./agent_memory.db` (default)

## Configuration

Configuration can be provided via a JSON file passed to the agent constructor:

```python
# config.json
{
    "planner": {
        "model_name": "qwen-code-3.5-30b"
    },
    "executor": {
        "timeout_seconds": 600
    }
}
```

## Example Tasks

### Dataset Build Task
```json
{
    "task_id": "build_dataset_1",
    "objective": "Build JWST dataset from FITS file",
    "inputs": {
        "input_fits": "/data/jwst_observation.fits"
    },
    "expected_outputs": [
        "output/index.csv",
        "output/manifest.json",
        "output/tiles/*.npy"
    ]
}
```

### Normalization Task
```json
{
    "task_id": "normalize_1",
    "objective": "Apply asinh normalization to image",
    "inputs": {
        "image_path": "/data/image.npy",
        "lower_percentile": 1.0,
        "upper_percentile": 99.8
    },
    "expected_outputs": [
        "output/normalized_image.npy"
    ]
}
```

## Validation Checks

The agent performs the following validations:
- File existence and integrity
- Manifest consistency with generated artifacts  
- Index.csv formatting and content
- Tile count verification
- Normalization sanity checks
- Flux conservation (placeholder)

## Troubleshooting

### Common Issues

1. **Command not allowed**: Check that your command is in the allowlist
2. **Validation failures**: Review validation reports for specific issues
3. **Memory errors**: Ensure sufficient disk space for storage

### Debugging

```python
# Get detailed run information
summary = agent.get_run_summary("task_id_123")
print(json.dumps(summary, indent=2, default=str))
```

## Development

The agent is designed to be extensible:
- Add new validators by extending `DatasetValidator`
- Add new planners by implementing the `Planner` interface
- Extend memory storage with additional tables or formats
- Add new command patterns to the allowlist

## License

This agent is part of the Astro-Flow-3D project and is released under the MIT license.