from pathlib import Path

from src.data.fits_loader import load_fits
from src.data.preprocess import normalize
from src.data.tile_generator import generate_tiles
from src.data.statistics import generate_statistics


def run_pipeline(
    fits_path,
    output_root="/data/astroflow/processed",
    tile_size=256,
    stride=256,
):
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Load FITS file
    2. Normalize science image
    3. Generate image tiles
    4. Compute tile statistics
    5. Return statistics DataFrame
    """

    fits_path = Path(fits_path)
    output_root = Path(output_root)

    tile_dir = output_root / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "index.csv"

    # Use FITS filename (without extension) as tile prefix
    source_name = fits_path.stem

    print("=" * 60)
    print("ASTRO-FLOW PREPROCESSING PIPELINE")
    print("=" * 60)

    print(f"\nInput FITS : {fits_path}")
    print(f"Output Dir : {output_root}")

    print("\n[1/4] Loading FITS...")
    image = load_fits(fits_path)

    print(f"Loaded image : {image.shape}")

    print("\n[2/4] Normalizing image...")
    image = normalize(image)

    print(
        f"Range : {image.min():.4f} → {image.max():.4f}"
    )

    print("\n[3/4] Generating tiles...")

    total_tiles = generate_tiles(
        image=image,
        output_dir=tile_dir,
        source_name=source_name,
        tile_size=tile_size,
        stride=stride,
    )

    print(f"Generated {total_tiles} tiles")

    print("\n[4/4] Computing statistics...")

    df = generate_statistics(
        tile_directory=tile_dir,
        csv_path=csv_path,
    )

    print("\nPipeline Complete")

    print(f"Tiles : {tile_dir}")
    print(f"CSV   : {csv_path}")

    print("=" * 60)

    return df


if __name__ == "__main__":

    FITS_FILE = (
        "downloads/mastDownload/JWST/"
        "jw01345003001_08201_00002_nrca2/"
        "jw01345003001_08201_00002_nrca2_i2d.fits"
    )

    run_pipeline(FITS_FILE)