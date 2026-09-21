## Learned

Quantify flux-ratio preservation of asinh vs percentile clipping

Hypothesis: A shared linear scale keeps band_A/band_B; independent [0,1] clipping does not. Asinh is a display stretch and is not ratio-preserving unless inverted.

Validation: shared linear median rel error=0.0000; shared asinh=0.0218; independent percentile=0.2048; tolerance=0.01

Delta: {'method': 'asinh stretch (Lupton / SDSS)', 'dataset': 'synthetic positive flux arrays standing in for JWST I2D MJy/sr', 'claimed_improvement': 'Preserve inter-band flux ratios that percentile min-max destroys', 'assumptions': ['Shared stretch parameter across bands', 'Linear flux in the unstretched SCI image'], 'pipeline_delta': 'Add a flux-preserving normalization path in src/data/preprocess.py before any second NIRCam filter is stacked as a channel.'}
