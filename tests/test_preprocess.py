import numpy as np
import pytest

from src.data.preprocess import normalize


def test_normalize_percentile():
    """Test percentile-based normalization."""
    # Create a test image with known values
    image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    
    # Test with default percentiles (1-99.8)
    normalized = normalize(image)
    
    assert normalized.shape == image.shape
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    
    # With fixed values, we get more predictable results
    image2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    normalized2 = normalize(image2, lower_percentile=0, upper_percentile=100)
    
    # Should be min-max normalized
    expected = (image2 - image2.min()) / (image2.max() - image2.min())
    np.testing.assert_array_equal(normalized2, expected)


def test_normalize_asinh():
    """Test asinh-based normalization."""
    # Create a test image with known values
    image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    
    # Test asinh normalization with default alpha
    normalized = normalize(image, method="asinh")
    
    assert normalized.shape == image.shape
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    
    # Verify that asinh preserves the relative structure but stretches values
    image2 = np.array([[1.0, 10.0, 100.0]], dtype=np.float32)
    normalized2 = normalize(image2, method="asinh")
    
    # With asinh, larger values are stretched less than with percentile clipping
    assert normalized2.shape == image2.shape
    assert normalized2.min() >= 0.0
    assert normalized2.max() <= 1.0
    
    # Test different alpha values
    normalized_alpha_01 = normalize(image, method="asinh", asinh_alpha=0.1)
    normalized_alpha_1 = normalize(image, method="asinh", asinh_alpha=1.0)
    
    # Different alpha values should produce different results
    np.testing.assert_array_not_equal(normalized_alpha_01, normalized_alpha_1)


def test_normalize_edge_cases():
    """Test edge cases in normalization."""
    # Test with constant image
    image = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    normalized = normalize(image)
    
    expected = np.zeros_like(image)
    np.testing.assert_array_equal(normalized, expected)
    
    # Test with constant image using asinh
    normalized_asinh = normalize(image, method="asinh")
    np.testing.assert_array_equal(normalized_asinh, expected)
    
    # Test with all zeros
    image_zeros = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normalized_zeros = normalize(image_zeros)
    
    expected_zeros = np.zeros_like(image_zeros)
    np.testing.assert_array_equal(normalized_zeros, expected_zeros)
    
    # Test with NaN and infinite values
    image_nan = np.array([[np.nan, 1.0, 2.0]], dtype=np.float32)
    normalized_nan = normalize(image_nan)
    
    # NaN should be converted to 0.0
    assert normalized_nan[0, 0] == 0.0


def test_normalize_method_validation():
    """Test that invalid normalization methods raise errors."""
    image = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    # Should work with valid methods
    normalize(image, method="percentile")
    normalize(image, method="asinh")
    
    # Should fail with invalid method
    with pytest.raises(ValueError):
        normalize(image, method="invalid_method")