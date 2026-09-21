# Quantify flux-ratio preservation of asinh vs percentile clipping

Hypothesis: A shared asinh stretch keeps band_A/band_B while independent [0,1] clipping does not.

Validation: On synthetic two-band flux, shared asinh median relative ratio error is within 1% and independent percentile clipping is worse.

Target files:
- src/data/preprocess.py

Do not push to main. Open a PR after scientific validation.
