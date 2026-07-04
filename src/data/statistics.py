from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BRIGHT_THRESHOLD = 0.80
DARK_THRESHOLD = 0.05


# ------------------------------------------------------------------
# Statistics for a single tile
# ------------------------------------------------------------------

def compute_statistics(tile: np.ndarray) -> dict:
    """
    Compute descriptive statistics for one normalized tile.
    """

    tile = np.nan_to_num(tile)

    mean = float(tile.mean())
    std = float(tile.std())
    minimum = float(tile.min())
    maximum = float(tile.max())
    median = float(np.median(tile))

    p99 = float(np.percentile(tile, 99))

    bright_pixels = int((tile > BRIGHT_THRESHOLD).sum())
    dark_pixels = int((tile < DARK_THRESHOLD).sum())

    nonzero_fraction = float((tile > 0).sum() / tile.size)

    dynamic_range = p99 - median

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "median": median,
        "p99": p99,
        "dynamic_range": dynamic_range,
        "bright_pixels": bright_pixels,
        "dark_pixels": dark_pixels,
        "nonzero_fraction": nonzero_fraction,
    }


# ------------------------------------------------------------------
# Quality Score
# ------------------------------------------------------------------

def compute_quality_score(stats: dict) -> float:
    """
    Compute a heuristic quality score.

    Used only for ranking tiles.
    """

    return float(
        stats["dynamic_range"] * stats["std"]
    )


# ------------------------------------------------------------------
# Filename Parser
# ------------------------------------------------------------------

def parse_tile_filename(filename: str) -> dict:
    """
    Parse tile filename.

    Supports both:

    New:
    jw01345003001_08201_00002_nrca2_i2d_tile_x00512_y01024_s256.npy

    Legacy:
    tile_0000.npy
    """

    stem = Path(filename).stem

    # -------------------------
    # New metadata-rich format
    # -------------------------

    pattern = (
        r"^(?P<source>.+)"
        r"_tile"
        r"_x(?P<x>\d+)"
        r"_y(?P<y>\d+)"
        r"_s(?P<size>\d+)$"
    )

    match = re.match(pattern, stem)

    if match:

        return {
            "source_fits": match.group("source"),
            "tile_x": int(match.group("x")),
            "tile_y": int(match.group("y")),
            "tile_size": int(match.group("size")),
        }

    # -------------------------
    # Legacy tiles
    # -------------------------

    legacy = re.match(
        r"^tile_(\d+)$",
        stem,
    )

    if legacy:

        return {
            "source_fits": "unknown",
            "tile_x": -1,
            "tile_y": -1,
            "tile_size": 256,
        }

    raise ValueError(
        f"Invalid tile filename: {filename}"
    )


# ------------------------------------------------------------------
# Build Statistics DataFrame
# ------------------------------------------------------------------

def build_statistics_dataframe(tile_directory):

    tile_directory = Path(tile_directory)

    tile_files = sorted(tile_directory.glob("*.npy"))

    rows = []

    for tile_file in tile_files:

        tile = np.load(tile_file)

        stats = compute_statistics(tile)

        stats["quality_score"] = compute_quality_score(stats)

        metadata = parse_tile_filename(
            tile_file.name
        )

        tile_uid = hashlib.sha1(
            tile_file.name.encode("utf-8")
        ).hexdigest()[:12]

        row = {
            "tile_uid": tile_uid,
            "tile_filename": tile_file.name,
        }

        row.update(metadata)
        row.update(stats)

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------

def save_statistics(
    dataframe: pd.DataFrame,
    csv_path,
):

    csv_path = Path(csv_path)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )


# ------------------------------------------------------------------
# Generate Statistics
# ------------------------------------------------------------------

def generate_statistics(
    tile_directory,
    csv_path,
):
    """
    Generate statistics and save to CSV.
    """

    df = build_statistics_dataframe(
        tile_directory
    )

    save_statistics(
        df,
        csv_path,
    )

    return df