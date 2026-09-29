# ViT Training Plan for Astro-Flow-3D

## The Problem We're Solving

**Scientific Challenge**: JWST observations span multiple filters (NIRCam: 0.6-5μm, MIRI: 4.9-28.8μm) with different spatial resolutions. MIRI's PSF is ~3× worse than NIRCam's at F444W. Current approaches:
- Treat MIRI as just another photometric data point
- Exclude MIRI from pixel-level analysis (resolution mismatch)
- Use classical SED fitting (hours per galaxy)

**Our Goal**: Build a **joint NIRCam+MIRI multimodal encoder** that learns spatially-resolved dust physics from both instruments simultaneously, despite the resolution mismatch.

---

## Model Architecture

### Option 1: ViT with Cross-Attention (Recommended)
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

**Why this works**:
- Cross-attention learns to upsample MIRI features using NIRCam context
- No need to pre-degrade NIRCam images (preserves resolution)
- End-to-end differentiable, no hand-designed kernels

### Option 2: CNN with Multi-Scale Fusion (Simpler)
```
NIRCam → [Conv2D × 6] → [ResNet-50 backbone] ──┐
                                              ├→ Concat → MLP → Output
MIRI  → [Conv2D × 4]  → [ConvNeXt backbone] ──┘
```

---

## Implementation Strategy

### Phase 1: Use Existing Pretrained ViT (Week 1-2)
**Do NOT train from scratch.** We have limited time.

| Component | Source | Why |
|-----------|--------|-----|
| Backbone | `timm` | Pretrained on ImageNet, MAE, or JAX-IM |
| Channel Adapter | 1×1 Conv | Inflates first layer to accept N bands |
| Decoder | Light MAE decoder | Reconstruct missing bands |

**Pretrained Backbones to Consider**:
```python
# From timm
vit_tiny_patch16_224_mae          # MAE pretrained
vit_small_patch16_224_dino        # DINO pretrained
swin_tiny_patch4_224              # Shifted Window, efficient
convnext_tiny_224                 # CNN alternative
```

### Phase 2: Fine-Tuning Strategy

**Stage 1: Masked Band Pretraining (Free Supervision)**
- Mask entire bands (not spatial patches)
- Train to predict masked band from others
- Loss: MSE between predicted and actual masked band

**Stage 2: Downstream Task Fine-Tuning**
- Load pretrained encoder
- Add task-specific head (MLP for regression)
- Fine-tune on labeled data (CEERS catalogs)

---

## Data Requirements

### Input Format
- **Tiles**: 256×256 patches from JWST images
- **Channels**: 6 NIRCam + 4 MIRI = 10 bands (or subset)
- **Output**: Physical parameters per tile

### Dataset Size Estimates
| Dataset | Tiles | GPU Hours |
|---------|-------|-----------|
| CEERS (1 tile) | ~10K | 1-2 |
| COSMOS-Web (10 tiles) | ~100K | 10-15 |
| Full JADES (100 tiles) | ~1M | 80-100 |

With **8×A100s** (30 hours), we can train on ~100K tiles:
- Batch size: 16 × 8 GPUs = 128
- Steps: 100K / 128 ≈ 780 steps/epoch
- epochs: ~20-30 for convergence

---

## Training Setup

### Distributed Configuration
```yaml
# configs/vit_training.yaml
model:
  type: "multimodal_vit"
  nircam_bands: ["F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
  miri_bands: ["F770W", "F1000W", "F1130W", "F1500W"]
  use_cross_attention: true

training:
  batch_size: 16  # Per GPU
  learning_rate: 1e-4
  epochs: 30
  warmup_steps: 1000
  weight_decay: 0.05
  
distributed:
  backend: "nccl"
  visible_devices: "0,1,2,3,4,5,6,7"
  grad_accumulation: 1
```

### Loss Functions
```python
# Masked band reconstruction
loss_recon = MSE(predicted_band, masked_band)

# Physical constraint (flux conservation)
loss_flux = L1(flux_predicted, flux_conserved)

# Total
loss = λ1 * loss_recon + λ2 * loss_flux
```

---

## What You Get After 30 Hours

| Time | Milestone |
|------|-----------|
| 0-6h | Pretrained encoder on masked bands |
| 6-12h | First validation run |
| 12-18h | Converged masked reconstruction |
| 18-24h | Fine-tuned on photometry labels |
| 24-30h | Generate dust maps for CEERS |

**Deliverables**:
1. `vit_multimodal_ceers.pt` - Pretrained model
2. `reconstruction_error_maps/` - NIRCam-dark candidates
3. `dust_mass_maps/` - Physical parameter maps
4. `metrics.jsonl` - Full training log

---

## Why This Justifies 8×A100s

| Component | GPU Demand |
|-----------|------------|
| ViT-S (22M params) | ~2GB VRAM |
| ViT-B (86M params) | ~5GB VRAM |
| Batch size 16 × 10 bands | ~12GB VRAM |
| **Total per GPU** | ~15GB |
| **8 GPUs** | 120GB total (plenty headroom) |

The 8 GPUs are used for:
- **Data parallelism**: 128 batch size (16 × 8)
- **Mixed precision**: Faster training
- **Checkpointing**: Parallel I/O
- **Validation**: Run evaluation while training continues

---

## Next Steps

1. **Pick a pretrained backbone** (timm provides ready-to-use models)
2. **Add channel adapter** for N-band inputs
3. **Implement masked band pretraining**
4. **Train on CEERS tiles** (30 hours on 8×A100)
5. **Validate reconstruction quality**

---

## References

- He et al. (2021) - Masked Autoencoders Are Scalable Vision Learners
- Robertson et al. (2023) - Morpheus: Galaxy morphology from JWST
- Cosmo et al. (2024) - COSMOS-Web photo-z with NIRCam+MIRI
- JADES Collaboration - Multi-wavelength SED fitting

---

*This plan is optimized for your 30-hour window on 8×A100 GPUs. It avoids training ViT from scratch and leverages existing pretrained weights.*
