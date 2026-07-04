import numpy as np


def normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
) -> np.ndarray:
    """
    Normalize a JWST science image using percentile clipping
    followed by min-max scaling.

    Parameters
    ----------
    image : np.ndarray
        Input science image.

    lower_percentile : float, default=1.0
        Lower clipping percentile.

    upper_percentile : float, default=99.8
        Upper clipping percentile.

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

    return image.astype(np.float32)