from pathlib import Path

import numpy as np

from src.data.statistics import parse_tile_filename


def load_tiles(tile_directory):
    """
    Load all tiles and their metadata from a directory.

    Parameters
    ----------
    tile_directory : str | Path

    Returns
    -------
    list[dict]
        Each element contains:
        - image
        - x
        - y
        - size
        - filename
        - source
    """

    tile_directory = Path(tile_directory)

    tile_files = sorted(tile_directory.glob("*.npy"))

    if len(tile_files) == 0:
        raise FileNotFoundError(
            f"No tiles found in {tile_directory}"
        )

    tiles = []

    for tile_file in tile_files:

        metadata = parse_tile_filename(tile_file.name)

        tiles.append(
            {
                "image": np.load(tile_file),
                "x": metadata["tile_x"],
                "y": metadata["tile_y"],
                "size": metadata["tile_size"],
                "filename": tile_file.name,
                "source": metadata["source_fits"],
            }
        )

    return tiles


def reconstruct_image(
    tiles,
    image_shape,
):
    """
    Reconstruct an image from tiles.

    Supports overlapping tiles through weighted averaging.

    Parameters
    ----------
    tiles : list[dict]

    image_shape : tuple[int, int]

    Returns
    -------
    np.ndarray
    """

    if len(tiles) == 0:
        raise ValueError("Tile list is empty.")

    height, width = image_shape

    accumulator = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weights = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for tile in tiles:

        image = tile["image"]

        x = tile["x"]
        y = tile["y"]

        size = tile["size"]

        accumulator[
            y:y + size,
            x:x + size,
        ] += image

        weights[
            y:y + size,
            x:x + size,
        ] += 1

    reconstructed = np.zeros_like(accumulator)

    mask = weights > 0

    reconstructed[mask] = (
        accumulator[mask] /
        weights[mask]
    )
    

    return reconstructed , weights