# Astro-Flow-3D Experiment Runbook

This document describes the persistent GPU training infrastructure for Astro-Flow-3D.

## Overview

The system supports running persistent GPU training experiments (~30 hours) with:
- Automatic checkpointing and resume
- GPU health monitoring and alerts
- Logging and provenance capture
- Graceful shutdown and artifact archiving
- Smoke tests for pre-flight validation

## Directory Structure

```
astro-flow-3d/
├── configs/experiments/          # Experiment configuration files
│   ├── persistent_run.yaml       # Main 30-hour run config
│   └── smoke_test.yaml           # Pre-flight validation config
├── src/experiments/
│   ├── checkpoint.py             # Checkpointing utilities
│   ├── logger.py                 # Logging utilities
│   ├── run.py                    # Training runner
│   ├── shutdown.py               # Controlled shutdown utilities
│   └── validator.py              # Validation utilities
├── scripts/
│   └── gpu_monitor.py            # GPU health monitoring
├── deploy/systemd/               # Systemd service files
│   └── astroflow-persistent.service
├── artifacts/experiments/        # Run artifacts
└── runbook.md                    # This file
```

## Quick Start

### 1. Run Smoke Tests (Pre-flight Validation)

```bash
python3 src/experiments/smoke_test.py
```

This validates:
- GPU accessibility
- Python imports (torch, numpy, etc.)
- Model loading and forward pass
- Checkpoint save/load
- Logging functionality

### 2. Configure the Experiment

Edit `configs/experiments/persistent_run.yaml`:

```yaml
run:
  id: "persistent_30h"
  duration_hours: 30

data:
  train_dataset: /data/astroflow/datasets/ceers_v1/train
  batch_size: 16
  num_workers: 8

training:
  steps: 50000
  learning_rate: 1e-4

checkpointing:
  save_frequency_minutes: 30
  max_checkpoints: 10

logging:
  log_dir: /data/astroflow/logs
```

### 3. Run the Experiment

#### Option A: Direct execution
```bash
python3 src/experiments/run.py --config configs/experiments/persistent_run.yaml
```

#### Option B: With systemd service

```bash
# Copy the service file
sudo cp deploy/systemd/astroflow-persistent.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Start the service
sudo systemctl start astroflow-persistent

# Monitor logs
sudo journalctl -u astroflow-persistent -f

# Stop the service
sudo systemctl stop astroflow-persistent
```

### 4. Monitor GPU Health

```bash
# One-time check
python3 scripts/gpu_monitor.py --single

# Continuous monitoring
python3 scripts/gpu_monitor.py --interval 60

# With custom threshold
python3 scripts/gpu_monitor.py --threshold-util 50 --interval 30
```

## Checkpointing

### Resume from Checkpoint

The runner automatically detects and resumes from the latest checkpoint:

```python
runner = TrainingRunner(config)
# Automatically resumes from latest checkpoint if present
result = runner.run()
```

### Manual Checkpoint Management

```python
from src.experiments.checkpoint import CheckpointManager

manager = CheckpointManager(save_dir="/data/astroflow/checkpoints")
latest = manager.get_latest_checkpoint()
resume_info = manager.resume_from_checkpoint(model, optimizer)
```

## Validation

Run validation on a checkpoint:

```bash
python3 src/experiments/validator.py --run-id <run_id> --artifacts-dir /data/astroflow
```

Validation checks:
- Checkpoint file exists and is loadable
- Model state is valid (no NaN/Inf)
- Forward pass works correctly
- Metrics file format is valid

## Artifacts

### Automatic Collection

Artifacts are collected after a run:
- Checkpoints (`checkpoint_step_*.pt`)
- Logs (`stdout_*.log`, `stderr_*.log`, `python_*.log`)
- Metrics (`metrics.jsonl`)
- Manifest (`manifest_<run_id>.json`)

### Manual Archiving

```python
from src.experiments.shutdown import ArtifactHandler

handler = ArtifactHandler("/data/astroflow", "/data/astroflow/artifacts")
handler.archive_artifacts("run_20260928_145000")
```

## Configuration Reference

### Run Config

| Field | Description |
|-------|-------------|
| `run.id` | Unique run identifier |
| `run.duration_hours` | Target duration for the run |

### Data Config

| Field | Description |
|-------|-------------|
| `data.train_dataset` | Path to training data |
| `data.batch_size` | Batch size per GPU |
| `data.num_workers` | Data loader workers |

### Training Config

| Field | Description |
|-------|-------------|
| `training.steps` | Total training steps |
| `training.learning_rate` | Initial learning rate |
| `training.weight_decay` | AdamW weight decay |

### Checkpoint Config

| Field | Description |
|-------|-------------|
| `checkpointing.save_frequency_minutes` | Save interval |
| `checkpointing.max_checkpoints` | Max checkpoints to keep |
| `checkpointing.save_optimizer_state` | Save optimizer state |
| `checkpointing.save_scheduler_state` | Save scheduler state |

### Logging Config

| Field | Description |
|-------|-------------|
| `logging.log_dir` | Directory for log files |
| `logging.log_level` | Logging verbosity |

## Graceful Shutdown

The runner handles SIGTERM/SIGINT for graceful shutdown:
1. Stops the training loop
2. Saves a final checkpoint
3. Closes all log files
4. Writes final manifest

## System Requirements

- Linux with NVIDIA drivers
- CUDA 12.x
- Python 3.12+
- PyTorch 2.13+
- 8x NVIDIA A100 80GB GPUs (configurable)

## Troubleshooting

### No GPUs Detected

```bash
nvidia-smi --query-gpu=name --format=csv
```

If this fails, check NVIDIA driver installation.

### Checkpoint Resume Fails

Check that the checkpoint file is not corrupted:
```bash
python3 -c "import torch; print(torch.load('checkpoint_step_1.pt', weights_only=False))"
```

### High GPU Memory Usage

Reduce batch size or enable gradient accumulation in the config.

## Files

- `configs/experiments/persistent_run.yaml` - Main experiment config
- `src/experiments/run.py` - Training runner
- `src/experiments/checkpoint.py` - Checkpoint utilities
- `src/experiments/logger.py` - Logging utilities
- `src/experiments/validator.py` - Validation utilities
- `src/experiments/shutdown.py` - Shutdown utilities
- `scripts/gpu_monitor.py` - GPU monitoring
