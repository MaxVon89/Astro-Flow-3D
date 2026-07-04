# Astro-Flow-3D

Astro-Flow-3D is a research framework for learning physically meaningful representations of galaxy morphology from James Webb Space Telescope (JWST) observations. The long-term objective is to develop a physics-aware deep learning system capable of probabilistic three-dimensional morpho-spectral reconstruction of galaxies from two-dimensional multi-band imaging.

The project combines astronomical image processing, modern computer vision, and deep learning within a reproducible software framework designed for large-scale JWST surveys.

---

## Scientific Motivation

The James Webb Space Telescope has fundamentally changed observational astronomy by revealing unprecedented detail in distant galaxies across multiple infrared wavelengths. Surveys such as CEERS, JADES, PRIMER, and COSMOS-Web now provide datasets that contain millions of galaxies spanning a broad range of evolutionary stages.

Despite these advances, most current computer vision pipelines treat astronomical observations as conventional natural images, often ignoring the physical structure and observational characteristics unique to astrophysical data.

Astro-Flow-3D explores an alternative direction by developing learning algorithms that preserve physically meaningful information throughout the processing pipeline while remaining scalable to modern astronomical surveys.

---

## Research Objectives

The project is organized around three primary goals.

First, establish a reproducible data engineering pipeline capable of transforming calibrated JWST observations into machine-learning-ready datasets.

Second, investigate deep learning architectures for galaxy detection, segmentation, morphology representation learning, and multi-task prediction.

Finally, develop a physics-aware probabilistic framework capable of reconstructing three-dimensional morpho-spectral galaxy representations from two-dimensional observations.

---

## Current Progress

The current development effort focuses on the data engineering foundation required for subsequent machine learning experiments.

The preprocessing pipeline currently supports:

- discovery and retrieval of JWST observations from MAST
- FITS inspection and scientific visualization
- percentile-based image normalization
- tile generation for large astronomical images
- statistical characterization of image tiles
- metadata extraction and dataset indexing
- PyTorch dataset generation
- reconstruction-based validation of preprocessing correctness

The reconstruction stage verifies that tiled observations can be reconstructed without introducing numerical artifacts, ensuring that downstream learning algorithms receive information-preserving inputs.

---

## Methodology

The current preprocessing workflow follows the sequence

```
JWST Observation
        │
        ▼
MAST Archive
        │
        ▼
Calibrated FITS Image
        │
        ▼
Scientific Preprocessing
        │
        ▼
Tile Generation
        │
        ▼
Statistical Characterization
        │
        ▼
Metadata Indexing
        │
        ▼
PyTorch Dataset
        │
        ▼
Deep Learning Models
```

The modular design allows each processing stage to be developed, tested, and validated independently while maintaining reproducibility across large observational datasets.

---

## Current Repository

```
src/
    data/
    models/
    training/
    utils/

scripts/
    data/

configs/
docker/
downloads/
notebooks/
```

Reusable functionality is implemented within `src/`, while executable workflows and experiments are maintained under `scripts/`.

---

## Research Roadmap

The project is being developed incrementally.

The current stage establishes the data engineering pipeline and preprocessing framework.

The next stage introduces dataset intelligence through multi-band alignment, multi-channel tensor generation, dataset versioning, augmentation strategies, and train-validation-test partitioning.

Subsequent stages investigate baseline convolutional architectures, transformer-based segmentation models, morphology representation learning, and ultimately probabilistic three-dimensional morpho-spectral reconstruction.

---

## Technologies

Astro-Flow-3D is implemented primarily in Python using

- PyTorch
- NumPy
- Pandas
- Astropy
- Astroquery
- Matplotlib

Development is fully containerized using Docker and integrated with GitHub Actions for continuous integration and reproducible execution.

---

## Research Philosophy

The primary objective of Astro-Flow-3D is not merely to train another segmentation model, but to establish a reusable research framework for astronomical machine learning.

Accordingly, the software emphasizes reproducibility, modularity, scientific correctness, and scalability over rapid experimentation. Every stage of the processing pipeline is designed as an independently testable component so that future work—including transformer architectures, physics-aware learning objectives, and probabilistic reconstruction—can be incorporated without requiring major architectural changes.