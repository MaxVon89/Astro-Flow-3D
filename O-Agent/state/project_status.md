# Live project status for agent intake (keep this in sync with the pipeline).

Current stage: single-band JWST I2D tiling. SCI HDU, percentile clip + min-max to [0, 1], 256×256 tiles, reconstruction check.

Not done: flux-preserving multi-band normalization, PSF homogenization, multi-channel tensors, masked-band pretraining, simulation-based labels.

Scientific meaning of "3D" here: per-pixel physical latents (SED-derived), not geometric depth.
