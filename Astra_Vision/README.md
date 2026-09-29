# Astra-Vision: JWST Multi-Modal Training Project

This project implements a joint NIRCam+MIRI multimodal encoder for JWST data analysis.

## Overview

**Goal**: Build a joint NIRCam+MIRI multimodal encoder that learns spatially-resolved dust physics from both instruments simultaneously, despite the resolution mismatch.

### Key Insights

- **MIRI's PSF is ~3× worse than NIRCam's** at F444W
- Most ML work treats MIRI as just another photometric data point
- **This project treats MIRI images as spatial data** and learns to bridge the resolution gap via cross-attention

### Architecture

```
NIRCam Bands (6 bands @ 0.03"/px)    MIRI Bands (4 bands @ 0.12"/px)
         │                                      │
    [Patch Embedding]                    [Patch Embedding]
         │                                      │
    ViT Encoder Blocks                    ViT Encoder Blocks
         │                                      │
    ┌────┴────┐                          ┌────┴────┐
    │         │                          │         │
Cross-Attention Layers (MIRI→NIRCa) → Fuse → Physical Parameters
         │
    [Latent: Dust Mass, SFR, AGN]
```

## Project Structure

```
Astra-Vision/
├── Plans/                      # Project plans and design docs
│   ├── JWST-Multi-Modal-Training-Plan.md
│   └── ViT-Training-Plan.md
├── data/                       # Data download and preprocessing
│   ├── download.py            # Download JWST data from MAST
│   ├── preprocess.py          # Multi-band preprocessing
│   └── dataset.py             # PyTorch Dataset for multi-band tiles
├── models/                     # Model architectures
│   ├── vit_multimodal.py      # Cross-attention ViT
│   └── cross_attention.py     # PSF-aware cross-attention layers
├── training/                   # Training scripts
│   ├── train.py               # Main training script
│   ├── mask.py                # Masked band pretraining
│   └── logger.py              # Metrics logging
├── configs/                    # Experiment configs
│   └── training.yaml
├── outputs/                    # Checkpoints and results
│   ├── checkpoints/
│   └── logs/
└── examples.py                 # Example usage scripts
```

## Data

### JWST Bands Used

**NIRCam** (6 bands @ 0.03"/px):
- F115W, F150W, F200W, F277W, F356W, F444W

**MIRI** (4 bands @ 0.12"/px):
- F770W, F1000W, F1130W, F1500W

### Data Source

JWST data is downloaded from the **MAST archive** (`mast.stsci.edu`).

## Installation

```bash
cd Astra-Vision
pip install -r requirements.txt
```

## Quick Start

### 1. Download JWST Data

```bash
# From Astra-Vision directory
python run_pipeline.py --stage download --download-dir downloads/ceers --bands nircam,miri
```

Or directly:
```bash
python data/download.py --field ceers --bands nircam,miri --output downloads
```

### 2. Preprocess Data

```bash
python run_pipeline.py --stage preprocess --input-dir downloads/ceers --output-dir data/processed
```

Or use the preprocessing module directly:
```python
from data.preprocess import load_multiband_fits, preprocess_multiband, create_tile_dataset

# Load multi-band FITS
multi_band, headers = load_multiband_fits({
    'nircam': 'path/to/nircam.fits',
    'miri': 'path/to/miri.fits'
})

# Preprocess
processed = preprocess_multiband(multi_band, band_fwhms={'nircam': 0.06, 'miri': 0.24})

# Create tiles
create_tile_dataset(processed, 'data/tiles', tile_size=256)
```

### 3. Train Model

```bash
python run_pipeline.py --stage train --data-dir data/processed/tiles --output outputs
```

Or manually:
```python
from models.vit_multimodal import MultimodalViT
from training.mask import MaskedBandPretrainer
from data.dataset import MultiBandTileDataset
from torch.utils.data import DataLoader

# Create model
model = MultimodalViT(nircam_bands=6, miri_bands=4)

# Create dataset and loader
dataset = MultiBandTileDataset(tile_dir='data/tiles')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train
pretrainer = MaskedBandPretrainer(model, mask_prob=0.5)
pretrainer.train_epoch(loader)
```

## Training Configuration

Edit `configs/training.yaml` to configure:
- Model variant (tiny, small, base, large)
- Learning rate, batch size, epochs
- PSF scale factor
- Cross-attention settings

### Available Model Variants

| Variant | Parameters | VRAM | Use Case |
|---------|------------|------|----------|
| tiny    | ~7M        | ~2GB | Quick experiments |
| small   | ~27M       | ~4GB | Balanced performance |
| base    | ~104M      | ~8GB | Production models |
| large   | ~440M      | ~16GB | Maximum capacity |

## Usage Examples

See `examples.py` for complete examples:
```bash
python examples.py --example pipeline
```

## Data Pipeline

1. **Download**: Use `data/download.py` to fetch JWST data from MAST
2. **Preprocess**: `data/preprocess.py` handles multi-band alignment and normalization
3. **Dataset**: `data/dataset.py` provides PyTorch Dataset with masking support
4. **Train**: `training/train.py` orchestrates training with checkpointing

## References

- He et al. (2021) - Masked Autoencoders Are Scalable Vision Learners
- Robertson et al. (2023) - Morpheus: Galaxy morphology from JWST
- Cosmo et al. (2024) - COSMOS-Web photo-z with NIRCam+MIRI
- JADES Collaboration - Multi-wavelength SED fitting

## License

This project is for research purposes within the Astro-Flow-3D project.
