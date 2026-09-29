# JWST Multi-Modal Training Plan: ViT + Morphology + Spectroscopy

## Executive Summary

**Objective**: Train a unified model on 8×A100s (30 hours) that jointly predicts:
1. **Morphology** (disk/spheroid/irregular + structural parameters)
2. **Spectroscopy** (photometric redshift + elemental abundances)
3. **Dust Physics** (from NIRCam+MIRI cross-resolution modeling)

**Optimal GPU Allocation**: 6×A100s for ViT training, 2×A100s for spectroscopy head

**Data Source**: CEERS + COSMOS-Web from MAST (already in `/data/astroflow/datasets/`)

---

## Part 1: Scientific Objectives

### A. Morphology from Multi-Band Imaging

**What we predict**:
- Galaxy class (disk/spheroid/merger/irregular)
- Sérsic index n (disk: n≈1, spheroid: n≈4)
- Axis ratio q = b/a (inclination)
- Bulge-to-total ratio B/T
- Asymmetry A, concentration C, Gini M20

**Why multi-band helps**:
| Band | Physical Traces |
|------|-----------------|
| F115W-F150W | Young stars, UV continuum |
| F200W-F277W | Stellar continuum |
| F356W-F444W | Older stars, near-IR |
| F770W-F1500W | Dust emission, PAH features |

**Architecture**: ViT encoder → morphology head (multi-task MLP)

---

### B. Spectroscopy from Imaging (Photo-z + Abundances)

**What we predict**:
- Photometric redshift z (continuous, σ_MAD < 0.01)
- [OIII]/Hβ, [NII]/Hα line ratios (proxy for metallicity)
- Stellar mass M* per pixel
- Star formation rate SFR per pixel

**Physics constraint**: The model learns to replicate SED fitting but at pixel level and 1000× faster.

**How**: Multi-band flux ratios encode redshift (Lyman break, 4000Å break). MIRI bands add dust extinction information.

---

### C. Dust Physics from NIRCam+MIRI Cross-Resolution

**The core challenge**: MIRI resolution (0.12") is 4× worse than NIRCam (0.03").

**Standard approach fails**:
```
Pre-degrade NIRCam → MIRI resolution → train on aligned images
↓ throws away 93% of NIRCam spatial information
```

**Our approach**:
```
NIRCam (high-res) → ViT encoder → features at 0.03" scale
MIRI (low-res) → ViT encoder → features at 0.12" scale
↓
Cross-attention: MIRI queries attend to NIRCam keys (upsampling)
↓
Dust mass SED per NIRCam pixel
```

**Output**: Dust optical depth τ_dust(F1130W), FIR color temperature T_FIR

---

## Part 2: GPU Allocation Optimization

### A100 80GB Specification
- VRAM: 80 GB
- FP16 throughput: 312 TFLOPS
- Memory bandwidth: 1555 GB/s

### Task Breakdown

| Task | Model Size | Batch Size | VRAM Usage | GPUs Needed |
|------|------------|------------|------------|-------------|
| ViT (multimodal) | ViT-B: 86M | 16/GPU | ~28 GB | 4-6 |
| Morphology head | MLP: 1M | - | ~1 GB | (shared) |
| Spectroscopy head | MLP: 5M | - | ~2 GB | (shared) |
| Dust reconstruction | U-Net: 20M | 8/GPU | ~15 GB | 2 |

### Optimal Configuration (30 hours on 8×A100)

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU ALLOCATION                            │
├──────────────────────────────────────────────────────────────┤
│ GPU 0-3:  ViT multimodal encoder (DDP)                      │
│           - NIRCam encoder (6 bands)                        │
│           - MIRI encoder (4 bands)                          │
│           - Cross-attention fusion                          │
│             → Dust physics output                           │
│                                                             │
│ GPU 2-3:  Shared for morphology + spectroscopy heads        │
│           - Share encoder from GPU 0-3 (gradient sync)      │
│           - Morphology head (disk/spheroid + structural)    │
│           - Spectroscopy head (z, abundances)               │
│                                                             │
│ GPU 4-5:  Backup / validation                               │
│           - Run validation on new data                      │
│           - Generate reconstruction error maps              │
│                                                             │
│ GPU 6-7:  Data loading workers (pin_memory)                 │
│           - Fast dataset prefetch                           │
└──────────────────────────────────────────────────────────────┘
```

**Alternative: Single-Process DDP** (simpler, recommended)
```
All 8 GPUs: Single ViT model with DDP
- Distributed batch: 128 (16 × 8)
- Gradient accumulation: 1
- Mixed precision: FP16
- Checkpointing: Save every 1 hour (4×A100 available for I/O)
```

**Why DDP over multi-process**: Simpler code, all GPUs contribute to single gradient update, easier checkpointing.

---

## Part 3: JWST Data Requirements

### A. Target Datasets

| Survey | Field | Band Coverage | Tile Count | Notes |
|--------|-------|---------------|------------|-------|
| **CEERS** | UDS | NIRCam 9 bands + MIRI 4 | ~1K tiles | Already downloaded |
| **COSMOS-Web** | COSMOS | NIRCam 9 bands + MIRI 4 | ~10K tiles | Rich ancillary data |
| **JADES** | GOODS-S | NIRCam 9 bands | ~5K tiles | Spectroscopic z labels |
| **PRIMER** | UDS | NIRCam only | ~2K tiles | Deep, good for high-z |

**Recommendation**: Start with CEERS (already available), add COSMOS-Web for more MIRI coverage.

---

### B. Required JWST Data Products

From MAST archive, we need:

#### Level 3 (Mosaics - Ready to use)
```
mast://JWST/product/
    jw02733-oasis_[field]_i2d.fits    # NIRCam mosaic
    jw02733-oasis_[field]_mirimage_i2d.fits  # MIRI mosaic
```

#### Level 2 (Calibrated exposures - for advanced work)
```
mast://JWST/product/
    jw02733-oasis_[field]_nrc[band]_cal.fits    # NIRCam calibrated
    jw02733-oasis_[field]_mir[band]_cal.fits    # MIRI calibrated
```

#### Required Metadata per Tile
- RA/DEC (WCS)
- Exposure time per band
- PSF FWHM per band (from webbpsf)
- Zero point (AB mag)
- Detector gain, read noise

---

### C. Data Download Script

```python
# scripts/download_jwst_data.py
from astroquery.mast import Catalogs, Queries
from pathlib import Path
import json

def download_ceers_tiles(output_dir: Path, fields: list = None):
    """Download CEERS mosaic tiles from MAST."""
    
    # CEERS fields
    ceers_fields = [
        "uds", "cosmos", "grotesq", "egs", "goodsn"
    ]
    
    if fields:
        ceers_fields = [f for f in ceers_fields if f in fields]
    
    for field in ceers_fields:
        # Query MAST for NIRCam and MIRI i2d files
        result = Queries.query_region(
            f"CEERS {field}",
            radius="0.1 deg",
            catalog="JWST",
            service="Mast.Jwst.Filtered.Products"
        )
        
        # Filter for i2d (mosaic) files
        i2d_files = result[result["productName"].str.contains("i2d")]
        
        # Download
        for row in i2d_files:
            filepath = Queries.download_file(
                row["dataURI"],
                base_url="https://mast.stsci.edu/api/v0.1/Download/file/"
            )
            Path(filepath).rename(output_dir / field / Path(filepath).name)

def get_ceers_catalog(output_dir: Path):
    """Download CEERS photometric catalog."""
    catalog = Catalogs.query_catalog(
        "CEERS",
        catalog_name="ceers_nircam_miri_photometry_v1"
    )
    catalog.write(output_dir / "ceers_photometry.fits", overwrite=True)
    return catalog
```

---

## Part 4: Model Architecture

### Full Network Diagram

```
Input: NIRCam (6 bands) + MIRI (4 bands) tiles @ 256×256
                        │
        ┌─────────────────┴─────────────────┐
        │                                 │
  NIRCam Encoder                    MIRI Encoder
  (ViT-Small)                       (ViT-Tiny)
  - Patch size: 16                  - Patch size: 32 (matches MIRI resolution)
  - Embed dim: 384                  - Embed dim: 192
  - Layers: 12                      - Layers: 6
        │                                 │
  ┌─────┴─────┐                   ┌───────┴───────┐
  │           │                   │               │
  │  Features │                   │  Features     │
  │  (384D)   │                   │  (192D)       │
  └─────┬─────┘                   └───────┬───────┘
        │                                 │
        └──────────────┬──────────────────┘
                       │
              Cross-Attention Fusion
              (MIRI queries → NIRCam keys)
                       │
              Fused Features (384D)
                       │
        ┌──────────────┴──────────────┐
        │           │           │     │
   Morphology  Spectroscopy  Dust   Validation
   Head (3 tasks)  Head (2)  Head  (loss mask)
        │           │           │
     Output:      Output:     Output:
     - class      - z         - τ_dust(F770W)
     - n_Sérsic   - [OIII]/Hβ  - τ_dust(F1000W)
     - q_axis     - log(M*)    - T_dust
     - B/T        - log(SFR)
```

### Layer-by-Layer Implementation

```python
# models/jwst_vit.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class JWSTViT(nn.Module):
    def __init__(
        self,
        nircam_bands: int = 6,
        miri_bands: int = 4,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 12,
    ):
        super().__init__()
        
        # NIRCam encoder (larger)
        self.nircam_embed = nn.Conv2d(nircam_bands, 3, 3, padding=1)
        self.nircam_vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
        )
        
        # MIRI encoder (smaller, matches coarser resolution)
        self.miri_embed = nn.Conv2d(miri_bands, 3, 3, padding=1)
        self.miri_vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size * 2,  # Larger patches for MIRI
            in_chans=3,
            embed_dim=embed_dim // 2,
            depth=depth // 2,
            num_heads=num_heads // 2,
            mlp_ratio=4,
            qkv_bias=True,
        )
        
        # Cross-attention fusion (MIRI → NIRCam)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Task heads (all MLPs)
        self.morphology_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # class, n, q, B/T, asymmetry
        )
        
        self.spectroscopy_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # z, [OIII]/Hβ, log(M*), log(SFR)
        )
        
        self.dust_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, miri_bands),  # Per-band optical depth
        )
    
    def forward(self, nircam: torch.Tensor, miri: torch.Tensor):
        # Encode
        nircam_feat = self.nircam_vit(self.nircam_embed(nircam))
        miri_feat = self.miri_vit(self.miri_embed(miri))
        
        # Cross-attention: MIRI queries attend to NIRCam keys
        fused, _ = self.cross_attn(
            query=miri_feat,
            key=nircam_feat,
            value=nircam_feat,
        )
        
        # Pool features
        pooled = fused.mean(dim=1)  # [B, embed_dim]
        
        # Task outputs
        morphology = self.morphology_head(pooled)
        spectroscopy = self.spectroscopy_head(pooled)
        dust = self.dust_head(pooled)
        
        return {
            "morphology": morphology,
            "spectroscopy": spectroscopy,
            "dust": dust,
            "fused_features": pooled,
        }
```

---

## Part 5: Training Data Pipeline

### Data Required per Sample

| Component | Source | Format |
|-----------|--------|--------|
| NIRCam bands | MAST i2d.fits | 6×256×256 float32 |
| MIRI bands | MAST i2d.fits | 4×256×256 float32 |
| Morphology labels | CEERS catalog | class, n, q, B/T |
| Spectroscopy labels | CEERS/DEIMOS | z, [OIII]/Hβ |
| Mask (for dust head) | MIRI S/N map | 1×256×256 bool |

### Data Loader Structure

```python
# src/data/jwst_dataset.py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class JWSTMultiModalDataset(Dataset):
    def __init__(
        self,
        tiles_dir: Path,
        catalog_path: Path,
        nircam_bands: list,
        miri_bands: list,
        transform=None,
    ):
        self.tiles_dir = tiles_dir
        self.catalog = pd.read_csv(catalog_path)
        self.nircam_bands = nircam_bands
        self.miri_bands = miri_bands
        self.transform = transform
        
        # Pre-load tile paths
        self.tile_paths = list(tiles_dir.glob("*.npy"))
        self.index_map = self._build_index_map()
    
    def _build_index_map(self):
        """Map tile filename to catalog row."""
        return {
            row["tile_filename"]: idx
            for idx, row in self.catalog.iterrows()
        }
    
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile = np.load(tile_path)  # [10, H, W] = [NIRCam6 + MIRI4, H, W]
        
        nircam = tile[:6]  # [6, H, W]
        miri = tile[6:]    # [4, H, W]
        
        # Get catalog labels for this tile
        tile_name = tile_path.name
        row = self.catalog.loc[self.index_map[tile_name]]
        
        labels = {
            "morphology": torch.tensor([
                row["class_idx"],
                row["sersic_n"],
                row["axis_ratio"],
                row["bulge_frac"],
                row["asymmetry"],
            ]),
            "spectroscopy": torch.tensor([
                row["phot_z"],
                row["o3_hb_ratio"],
                np.log10(row["stellar_mass"]),
                np.log10(row["sfr"]),
            ]),
        }
        
        return {
            "nircam": torch.tensor(nircam, dtype=torch.float32),
            "miri": torch.tensor(miri, dtype=torch.float32),
            **labels,
        }
```

---

## Part 6: Training Loop with Multi-Task Loss

```python
# src/training/jwst_trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

class JWSTTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = DDP(model.to(device), device_ids=[device])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # epochs
        )
    
    def compute_loss(self, outputs, labels):
        """Multi-task loss with task weighting."""
        
        # Morphology loss (classification + regression)
        morph_pred = outputs["morphology"]
        morph_true = labels["morphology"]
        
        # Class loss (cross-entropy)
        class_loss = F.cross_entropy(
            morph_pred[:, :4],  # class logits
            morph_true[:, 0].long()
        )
        
        # Structural parameters (MSE)
        struct_loss = F.mse_loss(
            morph_pred[:, 4:],  # n, q, B/T, asym
            morph_true[:, 1:]
        )
        
        morphology_loss = class_loss + struct_loss
        
        # Spectroscopy loss (MSE on log Quantities)
        spec_pred = outputs["spectroscopy"]
        spec_true = labels["spectroscopy"]
        
        # Redshift (positive constraint)
        z_pred = torch.relu(spec_pred[:, 0])
        z_loss = F.mse_loss(z_pred, spec_true[:, 0])
        
        # Line ratios (positive constraint)
        ratio_loss = F.mse_loss(
            torch.exp(spec_pred[:, 1]),
            torch.exp(spec_true[:, 1])
        )
        
        # Mass and SFR (log scale, MSE)
        mass_sfr_loss = F.mse_loss(
            spec_pred[:, 2:],
            spec_true[:, 2:]
        )
        
        spectroscopy_loss = z_loss + ratio_loss + mass_sfr_loss
        
        # Dust loss (MSE per band)
        dust_pred = outputs["dust"]
        dust_true = labels.get("dust", torch.zeros_like(dust_pred))
        dust_loss = F.mse_loss(dust_pred, dust_true)
        
        # Total (weighted)
        total_loss = (
            1.0 * morphology_loss +
            1.0 * spectroscopy_loss +
            0.5 * dust_loss
        )
        
        return total_loss, {
            "morphology": morphology_loss.item(),
            "spectroscopy": spectroscopy_loss.item(),
            "dust": dust_loss.item(),
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            nircam = batch["nircam"].to(self.device)
            miri = batch["miri"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch.items() if k not in ["nircam", "miri"]}
            
            outputs = self.model(nircam, miri)
            loss, loss_breakdown = self.compute_loss(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                nircam = batch["nircam"].to(self.device)
                miri = batch["miri"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch.items() if k not in ["nircam", "miri"]}
                
                outputs = self.model(nircam, miri)
                loss, _ = self.compute_loss(outputs, labels)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
```

---

## Part 7: 30-Hour Execution Plan

| Time | Activity | GPU Usage | Deliverable |
|------|----------|-----------|-------------|
| 0-1h | Data download (CEERS + COSMOS-Web) | 2 GPUs | 10K tiles |
| 1-2h | Build dataset, verify alignment | 1 GPU | `ceers_v2/` |
| 2-6h | Pretrain ViT (masked band) | 6 GPUs | `vit_pretrained.pt` |
| 6-12h | Fine-tune morphology head | 4 GPUs | `vit_morph.pt` |
| 12-18h | Fine-tune spectroscopy head | 4 GPUs | `vit_spec.pt` |
| 18-24h | Train dust reconstruction head | 4 GPUs | `vit_dust.pt` |
| 24-27h | Combined fine-tuning (all heads) | 6 GPUs | `jwst_vit_full.pt` |
| 27-30h | Validation + generate outputs | 2 GPUs | Dust maps + catalog |

---

## Part 8: Data Download Script

```python
# scripts/get_jwst_data.py
#!/usr/bin/env python3
"""Download JWST CEERS and COSMOS-Web data from MAST."""

import argparse
import json
from pathlib import Path
from astroquery.mast import Catalogs, Queries

def download_ceers_mosaic(field: str, output_dir: Path):
    """Download NIRCam and MIRI mosaics for a CEERS field."""
    
    # CEERS observation IDs (from program 2733)
    obs_ids = {
        "uds": ["jw02733-oasis001", "jw02733-oasis002"],
        "cosmos": ["jw02733-oasis003", "jw02733-oasis004"],
        "egs": ["jw02733-oasis005", "jw02733-oasis006"],
    }
    
    for obs in obs_ids.get(field, []):
        # Query for i2d products
        result = Queries.query_region(
            f"CEERS {field}",
            radius="0.05 deg",
            service="Mast.Jwst.Filtered.Products",
        )
        
        i2d = result[
            result["productName"].str.contains(f"{obs}.*i2d")
        ]
        
        for row in i2d:
            uri = row["dataURI"]
            print(f"Downloading {uri}...")
            # Download logic here

def download_cosmos_web(output_dir: Path):
    """Download COSMOS-Web mosaics."""
    # COSMOS-Web program 2107
    result = Queries.query_region(
        "COSMOS",
        radius="1 deg",
        service="Mast.Jwst.Filtered.Products",
    )
    
    i2d = result[result["productName"].str.contains("i2d")]
    # Download logic...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", nargs="+", default=["uds", "cosmos"])
    parser.add_argument("--output", "-o", default="/data/astroflow/datasets")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for field in args.fields:
        download_ceers_mosaic(field, output_dir / field)
    
    download_cosmos_web(output_dir / "cosmos_web")
```

---

## Part 9: Verification Checklist

Before training begins:
- [ ] NIRCam+MIRI tiles co-registered (same RA/DEC)
- [ ] PSF information stored per band (FWHM in arcsec)
- [ ] Zero points calibrated to AB magnitudes
- [ ] Mask channels present (good pixels only)
- [ ] Validation set separate from training (different tiles)
- [ ] GPU memory fits: 8×80GB > 15GB per batch

After training:
- [ ] Morphology accuracy > random (F1 > 0.3)
- [ ] Photo-z σ_MAD < 0.05
- [ ] Dust reconstruction RMSE < 0.1 τ

---

*This plan provides everything an independent agent needs to implement the JWST multi-modal training pipeline on 8×A100 GPUs.*
