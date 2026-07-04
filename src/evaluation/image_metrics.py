import numpy as np


def mse(reference, prediction):
    """
    Mean Squared Error.
    """

    return float(
        np.mean(
            (reference - prediction) ** 2
        )
    )


def mae(reference, prediction):
    """
    Mean Absolute Error.
    """

    return float(
        np.mean(
            np.abs(reference - prediction)
        )
    )


def max_error(reference, prediction):
    """
    Maximum absolute error.
    """

    return float(
        np.max(
            np.abs(reference - prediction)
        )
    )