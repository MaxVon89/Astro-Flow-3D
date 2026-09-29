# Experiment Explanation: 30-Hour GPU Training Run

## What This Experiment Does

The command `python3 src/experiments/run.py --configs configs/experiments/persistent_run.yaml` starts a **persistent 30-hour GPU training run** for the Astro-Flow-3D project. This is not a one-off experiment—it's designed to run for days, with checkpointing to survive interruptions.

---

## How This Helps Achieve the Plans

The [Plans/](../Plans/) directory outlines a **4-day, 8x A100 GPU roadmap**. This experiment directly supports **Day 1 and Day 2** of that plan:

| Plan Goal | How This Experiment Delivers |
|-----------|-------------------------------|
| **Train the Masked Band Autoencoder (MBA)** | The `run.py` script executes a PyTorch training loop that trains on multi-band astronomical images (F115W, F200W, F356W, F444W) |
| **Learn inter-band physical relationships** | The MBA predicts F444W from the other three bands—learning physics from the data structure itself |
| **Build a self-validating system** | The experiment logs reconstruction error, which directly becomes the metric for NIRCam-dark detection |
| **Produce a NIRCam-dark catalog** | The MBA's reconstruction error identifies sources that appear in MIRI (F444W) but not NIRCam |

### Why This Is the Right First Step

From the Plans:
> *"It's the only architecture that produces a novel scientific result without labels. You have no ground-truth 3D galaxy structures... But you have multi-band images. The MBA turns the band structure itself into the supervision signal."*

This experiment:
1. **Runs 50,000 training steps** on 8x A100 GPUs (~30 hours)
2. **Saves checkpoints every 30 minutes** so training can resume after interruptions
3. **Logs all metrics** (loss, GPU memory, validation metrics) to JSON for later analysis

---

## Key Components of the System

### 1. `run.py` - The Training Runner
**Location:** `src/experiments/run.py`

This script manages the entire training lifecycle:

| Component | Purpose |
|-----------|---------|
| `CheckpointManager` | Saves model state every 30 min; supports resuming from any checkpoint |
| `MetricsLogger` | Writes `metrics.jsonl` with step, loss, learning rate, GPU memory |
| `ExperimentLogger` | Captures stdout/stderr; writes manifest with git SHA, GPU info, config |
| `TrainingRunner.run()` | Main loop: batches → forward pass → loss → backward pass → checkpoint |

**Current Model Architecture (placeholder):**
```python
model = nn.Sequential(
    nn.Conv2d(4, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 4, 3, padding=1),
)
```
*Note: This is a placeholder. The real MBA architecture needs to be substituted here.*

---

### 2. `persistent_run.yaml` - Configuration File
**Location:** `configs/experiments/persistent_run.yaml`

Key settings:
```yaml
data:
  batch_size: 16           # 16 images per GPU × 8 GPUs = 128 batch size
  num_workers: 8           # Parallel data loading

training:
  steps: 50000             # Total training steps (~30 hours on 8x A100)
  learning_rate: 0.0001    # AdamW optimizer
  visible_devices: "0,1,2,3,4,5,6,7"  # All 8 GPUs

checkpointing:
  save_frequency_minutes: 30   # Save every 30 minutes
  max_checkpoints: 10          # Keep only recent 10 checkpoints

validation:
  interval_steps: 500          # Run validation every 500 steps
```

---

## What We're Training the GPU For

### The Scientific Goal

**Detect NIRCam-dark galaxies using reconstruction error from an unsupervised MBA.**

| Component | What It Is | Why It Matters |
|-----------|------------|----------------|
| **MBA Encoder** | Learns to compress multi-band images into a latent representation | Captures physical relationships between bands (e.g., redshift, stellar population) |
| **MBA Decoder** | Reconstructs F444W from F115W+F200W+F356W | Poor reconstruction = missing NIRCam emission = high-redshift candidate |
| **Reconstruction Error** | Pixel-wise difference between predicted and actual F444W | The *signal* for NIRCam-dark detection; no labels needed |

### Training Pipeline

```
1. Load 4-band images (F115W, F200W, F356W, F444W) from CEERS dataset
2. Mask F444W band (treat as "missing")
3. Feed F115W+F200W+F356W to encoder
4. Decode to predict F444W
5. Compute loss = MSE(predicted_F444W, actual_F444W)
6. Backpropagate → update weights
7. Save checkpoint every 30 min
```

### What Gets Logged

**`/data/astroflow/logs/metrics.jsonl`:**
```json
{"step": 1000, "phase": "train", "loss": 0.0234, "learning_rate": 0.0001, "batch_time_ms": 150}
{"step": 1000, "phase": "validation", "val_loss": 0.0187, "val_rmse": 0.137}
```

**`/data/astroflow/logs/manifest_<run_id>.json`:**
- Git commit SHA
- GPU types and memory usage
- Full config used
- Timestamp

**`/data/astroflow/checkpoints/`:**
- `checkpoint_step_5000.pt` - Full model state (can resume from any step)

---

## How This Fits in the Full Plan

From **Plans/Day1.md**:

| Day | Goal | This Experiment's Role |
|-----|------|------------------------|
| Day 1 | Build MBA training infrastructure | ✅ `run.py` + checkpointing system |
| Day 2 | Train MBA on CEERS data | ✅ 50,000 steps = main training |
| Day 3 | Validate MBA reconstruction quality | Uses `checkpoint_step_*.pt` from this run |
| Day 4 | Generate NIRCam-dark catalog | Uses reconstruction error from trained MBA |

---

## Monitoring the Training

### Check GPU Usage
```bash
nvidia-smi --loop=1
```

### View Metrics in Real-Time
```bash
tail -f /data/astroflow/logs/metrics.jsonl | jq .
```

### Check Checkpoint Progress
```bash
ls -lh /data/astroflow/checkpoints/
```

### Resume from Checkpoint
The script auto-loads the latest checkpoint on restart:
```bash
python3 src/experiments/run.py --configs configs/experiments/persistent_run.yaml
```

---

## Troubleshooting

### No GPU Found
```bash
# Check CUDA is working
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU count
python3 -c "import torch; print(torch.cuda.device_count())"
```

### Out of Memory
Reduce `batch_size` in `persistent_run.yaml`:
```yaml
data:
  batch_size: 8  # Reduce from 16
```

### Interrupted Training
The system automatically resumes from the latest checkpoint—just re-run the same command.

---

## Next Steps After This Run

1. **Evaluate the trained MBA** on validation set
2. **Compute reconstruction error** for all CEERS sources
3. **Identify NIRCam-dark candidates** (high error in NIRCam bands)
4. **Cross-match** with existing catalogs to confirm novelty
5. **Generate final catalog** with positions, fluxes, error metrics

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/experiments/run.py` | Main training runner with checkpointing |
| `src/experiments/checkpoint.py` | Checkpoint management utilities |
| `src/experiments/logger.py` | Metrics logging (JSONL format) |
| `configs/experiments/persistent_run.yaml` | 30-hour training configuration |
| `configs/experiments/smoke_test.yaml` | Quick 2-step validation config |
| `n_agent/GUIDE-EXPERIMENT-RUNNER.md` | This file |
| `n_agent/30H-plan.md` | Detailed 30-hour timeline |
| `Plans/` | 4-day scientific roadmap |
