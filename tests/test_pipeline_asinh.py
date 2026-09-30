import tempfile
import numpy as np
import pytest
from pathlib import Path

from src.data.pipeline import run_pipeline
from src.data.preprocess import normalize


def test_pipeline_with_asinh_normalization():
    """
    Test that the full pipeline works with asinh normalization.
    """
    # Create a simple test image
    test_image = np.random.rand(100, 100).astype(np.float32)
    
    # Write test image to temporary file for the pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_fits_path = tmp_path / "test.fits"
        
        # Mock a FITS-like structure in numpy array format
        np.save(test_fits_path, test_image)
        
        # Test with percentile normalization
        output_dir1 = tmp_path / "output_percentile"
        df1 = run_pipeline(
            fits_path=test_fits_path,
            output_root=output_dir1,
            tile_size=32,
            stride=32,
            normalization_method="percentile"
        )
        
        # Test with asinh normalization
        output_dir2 = tmp_path / "output_asinh"
        df2 = run_pipeline(
            fits_path=test_fits_path,
            output_root=output_dir2,
            tile_size=32,
            stride=32,
            normalization_method="asinh"
        )
        
        # Both should return DataFrames with valid structure
        assert df1 is not None
        assert df2 is not None
        assert len(df1) > 0
        assert len(df2) > 0


def test_asinh_normalization_values():
    """
    Test that asinh normalization produces expected results.
    """
    # Create test image with known values
    test_image = np.array([[0.0, 1.0, 10.0, 100.0]], dtype=np.float32)
    
    # Test with percentile method
    normalized_percentile = normalize(test_image, method="percentile")
    
    # Test with asinh method
    normalized_asinh = normalize(test_image, method="asinh")
    
    # Both should be in [0,1] range
    assert normalized_percentile.min() >= 0.0
    assert normalized_percentile.max() <= 1.0
    assert normalized_asinh.min() >= 0.0
    assert normalized_asinh.max() <= 1.0
    
    # Results should be different (asinh is non-linear)
    np.testing.assert_array_not_equal(normalized_percentile, normalized_asinh)


if __name__ == "__main__":
    test_pipeline_with_asinh_normalization()
    test_asinh_normalization_values()
    print("All tests passed!")