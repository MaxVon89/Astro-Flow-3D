# Astro-Flow-3D Experiment Runner Guide

## What This Is

This is a **persistent GPU training infrastructure** for running long experiments (up to ~30 hours) on the Astro-Flow-3D neural network. It's designed to run unattended with automatic recovery from failures.

### Key Features

| Feature | What it does |
|---------|--------------|
| **Checkpointing** | Saves model state every 30 minutes; can resume from where it left off after crashes/reboots |
| **Monitoring** | Tracks GPU temperature, memory, utilization; alerts if GPUs go idle |
| **Logging** | Records all metrics (loss, learning rate) in machine-readable JSON format |
| **Provenance** | Captures git commit, config, CUDA version, environment for reproducibility |
| **Graceful Shutdown** | Handles SIGTERM/SIGINT to save final checkpoint before stopping |

### File Structure

```
astro-flow-3d/
├── configs/experiments/
│   ├── persistent_run.yaml    ← Main config for 30-hour runs
│   └── smoke_test.yaml        ← Quick validation config
├── src/experiments/
│   ├── run.py                 ← The training loop (main worker)
│   ├── checkpoint.py          ← Save/resume model state
│   ├── logger.py              ← Metrics and manifest generation
│   ├── validator.py           ← Validate checkpoints
│   ├── shutdown.py            ← Cleanup and archiving
│   ├── smoke_test.py          ← Pre-flight checks
│   └── launcher.py            ← Orchestrates everything
├── scripts/
│   └── gpu_monitor.py         ← Watch GPU health
└── EXPERIMENT_RUNBOOK.md      ← This documentation
```

---

## How to Use

### 1. Run Smoke Tests (First Time Setup)

```bash
cd /home/nvidia/projects/Astro-Flow-3D
python3 src/experiments/smoke_test.py
```

Expected output:
```
SMOKE TEST RESULTS
============================================================
Overall Status: pass

✓ gpu_access: pass      # GPUs are visible and working
✓ python_imports: pass  # All required packages loaded
✓ entrypoint: pass      # Model loads and runs
✓ checkpointing: pass   # Can save and load checkpoints
✓ logging: pass         # Logs are being written
✓ dataset_access: pass  # Dataset paths exist
```

### 2. Configure Your Experiment

Edit `configs/experiments/persistent_run.yaml`:

```yaml
data:
  train_dataset: /data/astroflow/datasets/your_dataset  # ← Change this
  batch_size: 16

training:
  steps: 50000         # How many steps to run
  learning_rate: 0.0001

checkpointing:
  save_frequency_minutes: 30   # Save every 30 minutes
```

### 3. Run the Experiment

**Option A: Direct execution (simplest)**
```bash
python3 src/experiments/run.py --config configs/experiments/persistent_run.yaml
```

**Option B: Using the launcher (recommended)**
```bash
python3 src/experiments/launcher.py \
  --config configs/experiments/persistent_run.yaml \
  --duration 30 \
  --no-monitor
```

### 4. Monitor GPU Health (Optional)

In another terminal:
```bash
python3 scripts/gpu_monitor.py --interval 60
```

This shows:
- GPU utilization (%)
- Temperature (°C)
- Memory usage (MB)
- Alerts if GPUs go idle for too long

### 5. Checkpointing

- Checkpoints saved to `/data/astroflow/checkpoints/checkpoint_step_*.pt`
- Automatically resumes from latest checkpoint on restart
- Keep up to 10 most recent checkpoints

### 6. Viewing Logs

```bash
# View recent metrics
tail -f /data/astroflow/logs/metrics.jsonl

# View stdout
tail -f /data/astroflow/logs/stdout_*.log
```

---

## Handling Crashes or Reboots

The system automatically handles this:

1. When you restart the experiment, it looks for the latest checkpoint
2. It resumes from that step (not from step 0)
3. No manual intervention needed

```bash
# Just run the same command - it will auto-resume
python3 src/experiments/run.py --config configs/experiments/persistent_run.yaml
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Validate setup | `python3 src/experiments/smoke_test.py` |
| Run 100-step test | `python3 src/experiments/run.py --config configs/experiments/smoke_test.yaml --steps 100` |
| Run 30-hour experiment | `python3 src/experiments/run.py --config configs/experiments/persistent_run.yaml` |
| Check GPU status | `python3 scripts/gpu_monitor.py --single` |
| Monitor GPU live | `python3 scripts/gpu_monitor.py --interval 60` |

---

## What's Happening Under the Hood

```
[Training Loop] 
    ↓
[Every 1 step] → Calculate loss, update weights
    ↓
[Every 30 min] → Save checkpoint to disk (atomic write)
    ↓
[Every 500 steps] → Run validation on holdout data
    ↓
[Every 60 sec] → Log metrics to JSON file
    ↓
[On SIGTERM] → Save final checkpoint, close logs gracefully
```

The key insight: **everything is designed to survive interruption**. If the machine reboots, the next run will find the last checkpoint and continue where it left off.
