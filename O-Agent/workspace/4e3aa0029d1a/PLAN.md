# Quantify flux-ratio preservation of asinh vs percentile clipping

Hypothesis: A shared linear scale keeps band_A/band_B; independent [0,1] clipping does not. Asinh is a display stretch and is not ratio-preserving unless inverted.

Validation: On synthetic two-band flux, shared linear median relative ratio error is within 1% and independent percentile clipping is worse.

Target files:
- src/data/preprocess.py

Do not push to main. Open a PR after scientific validation.
