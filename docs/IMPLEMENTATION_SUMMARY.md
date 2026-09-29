# Astro-Flow-3D - Asinh Normalization Implementation

## Summary of Changes

I've implemented the requested asinh normalization functionality in the preprocessing pipeline according to the research directions from Cl-1.md. 

### Key Updates

1. **Enhanced `normalize()` function** (in `src/data/preprocess.py`)
   - Added support for "asinh" normalization method
   - Maintained backward compatibility with existing "percentile" method
   - Added parameter `asinh_alpha` to control the stretch effect
   - Preserves physically meaningful inter-band ratios which is crucial for multi-band analysis

2. **Updated pipeline** (in `src/data/pipeline.py`)
   - Added new parameters `normalization_method` and `asinh_alpha`
   - Updated function signature to pass through these new normalization parameters

3. **Configuration support** (in `configs/preprocessing.yaml`)
   - Added normalization_method and asinh_alpha parameters
   - Initialized with defaults that maintain backward compatibility

4. **Dataset builder integration** (in `src/data/dataset_builder.py`)
   - Updated constructor to accept the new normalization parameters
   - Updated manifest.json generation to include normalization method information
   - Updated dataset summary output to show normalization method

5. **Unit tests** (in `tests/test_preprocess.py`)
   - Added comprehensive test suite for both percentile and asinh normalization methods
   - Tests cover edge cases like constant arrays, NaN values, and invalid inputs

6. **Integration tests** (in `tests/test_pipeline_asinh.py`)
   - Tests that validate the full pipeline with asinh normalization works correctly
   - Verifies that different normalization methods produce different results but consistent ranges

## Research Alignment

This change directly addresses research direction "A" from Cl-1.md ("Physically Meaningful Normalization"):
- "Look into asinh stretching (used in SDSS) and its physical motivation"
- "What normalization preserves the ratios between bands, since those ratios are what encode physical information (stellar temperature, dust reddening, redshift)?"

The asinh stretch is particularly beneficial because:
1. It preserves the relative structure of data while giving more reasonable stretching for large values
2. It's a common approach in astronomical image processing (SDSS)
3. It handles flux ratios more appropriately than simple percentile clipping
4. The alpha parameter allows fine-tuning of the stretch to match specific astronomical characteristics

## Backward Compatibility

All existing functionality remains intact:
- Default parameters maintain backward compatibility 
- Existing pipelines using percentile normalization continue to work unchanged
- Only new configurations that explicitly specify `normalization_method: asinh` will use the new method