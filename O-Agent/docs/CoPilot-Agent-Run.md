## 1. Run locally on the RTX 3060

Your laptop can run the early Astra experiments:

- FITS preprocessing and tiling
- 2-band experiments
- Small masked autoencoders
- Unit tests and scientific validation
- The LangGraph planner and local executor

It is not suitable for large training runs because the RTX 3060 has only 6 GB VRAM. Use small batches, 256×256 tiles, mixed precision, and gradient accumulation.

### Recommended setup

Use **WSL2 with Ubuntu** rather than native Windows Python.

Inside WSL2:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build from the official PyTorch selector. Then verify:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

You should see `True` and an RTX 3060.

In the training code, select the device explicitly:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

For training:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(batch)
    loss = criterion(output, target)
```

Start approximately with:

```text
tile size:       256
batch size:      1 or 2
mixed precision: enabled
workers:         2 or 4
gradient accumulation: 4 to 16
```

The agent itself does not need the GPU. Its planner, memory writer, validator, and orchestration code can run on the Ryzen CPU. Only the experiment runner should use CUDA.

## 2. Run on Brev

Use Brev for the expensive training jobs, not necessarily for the entire agent. The clean architecture is:

```text
Local machine: planner, code, tests, experiment definitions
Brev instance: GPU training and validation
Storage:       Git repository plus experiment artifacts
```

### Basic Brev workflow

1. Create a Brev instance with an NVIDIA GPU and an Ubuntu/PyTorch image.
2. Connect through SSH or the Brev-provided terminal.
3. Verify the GPU:

```bash
nvidia-smi
```

4. Clone the project:

```bash
git clone <your-repository-url>
cd Astra-Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

5. Install the project dependencies, then verify PyTorch:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

6. Run a training experiment:

```bash
python scripts/train.py \
  --device cuda \
  --batch-size 8 \
  --epochs 10 \
  --data /workspace/data
```

The exact Brev creation command depends on the current Brev CLI and available GPU templates, so use the command shown by your Brev dashboard or current Brev documentation.

### Important Brev considerations

- Put datasets and checkpoints on persistent storage; do not rely only on the instance disk.
- Store W&B, GitHub, and cloud credentials as Brev secrets or environment variables.
- Do not let the agent push directly to `main`; have it create a branch and pull request.
- Shut down the instance when idle because GPU billing usually continues while it is running.
- Use the local RTX 3060 for development and smoke tests; use Brev for full training and parameter sweeps.

A sensible first target is: develop locally with a tiny 2-band dataset, then submit the same training script to Brev unchanged with only the dataset path, batch size, and GPU configuration overridden.