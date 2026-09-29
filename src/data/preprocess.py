import numpy as np


def normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
    method: str = "percentile",
    asinh_alpha: float = 0.1,
) -> np.ndarray:
    """
    Normalize a JWST science image using various normalization methods.

    Parameters
    ----------
    image : np.ndarray
        Input science image.

    lower_percentile : float, default=1.0
        Lower clipping percentile for percentile method.

    upper_percentile : float, default=99.8
        Upper clipping percentile for percentile method.

    method : str, default="percentile"
        Normalization method to use: "percentile" or "asinh".

    asinh_alpha : float, default=0.1
        Alpha parameter for asinh normalization (controls stretch).

    Returns
    -------
    np.ndarray
        Normalized float32 image with values in [0, 1].
    """

    # Replace NaN and infinite values
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if method == "percentile":
        # Percentile clipping
        p_low = np.percentile(image, lower_percentile)
        p_high = np.percentile(image, upper_percentile)

        image = np.clip(image, p_low, p_high)

        # Min-max normalization
        min_val = image.min()
        max_val = image.max()

        if max_val == min_val:
            return np.zeros_like(image, dtype=np.float32)

        image = (image - min_val) / (max_val - min_val)
        
    elif method == "asinh":
        # Asinh stretch
        # Based on SDSS implementation
        image = np.arcsinh(image / asinh_alpha)
        
        # Min-max normalization to [0, 1]
        min_val = image.min()
        max_val = image.max()

        if max_val == min_val:
            return np.zeros_like(image, dtype=np.float32)

        image = (image - min_val) / (max_val - min_val)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return image.astype(np.float32)