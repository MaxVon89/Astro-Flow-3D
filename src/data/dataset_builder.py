import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd

from src.data.pipeline import run_pipeline


class DatasetBuilder:
    """Orchestrate dataset construction for JWST observations."""

    def __init__(
        self,
        input_root: Union[str, Path],
        output_root: Union[str, Path],
        tile_size: int = 256,
        stride: int = 256,
    ) -> None:
        """Create a dataset builder with input and output roots."""
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.tile_size = tile_size
        self.stride = stride

        if self.tile_size <= 0 or self.stride <= 0:
            raise ValueError("tile_size and stride must be positive integers")

        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

    def discover_fits(self) -> List[Path]:
        """Recursively search the input root for FITS files."""
        fits_files = sorted(self.input_root.rglob("*.fits"))

        if not fits_files:
            raise FileNotFoundError(
                f"No FITS files found under input root: {self.input_root}"
            )

        return fits_files

    def process_observation(self, fits_path: Union[str, Path]) -> pd.DataFrame:
        """Process a single FITS observation through the preprocessing pipeline."""
        fits_path = Path(fits_path)

        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        observation_output = self.output_root / fits_path.stem
        observation_output.mkdir(parents=True, exist_ok=True)

        return run_pipeline(
            fits_path=fits_path,
            output_root=observation_output,
            tile_size=self.tile_size,
            stride=self.stride,
        )

    def merge_indices(self) -> Path:
        """Merge all generated index.csv files into a single master index."""
        master_index = self.output_root / "index.csv"
        index_files = [
            path
            for path in sorted(self.output_root.rglob("index.csv"))
            if path != master_index
        ]

        if not index_files:
            raise FileNotFoundError(
                f"No generated index.csv files found under output root: {self.output_root}"
            )

        frames = [pd.read_csv(path) for path in index_files]
        merged = pd.concat(frames, ignore_index=True)

        merged.to_csv(master_index, index=False)

        return master_index

    def build_manifest(self) -> Path:
        """Write dataset metadata to manifest.json."""
        tile_paths = list(self.output_root.rglob("*.npy"))
        manifest = {
            "dataset_name": self.output_root.name,
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "num_images": len([item for item in self.output_root.iterdir() if item.is_dir()]),
            "num_tiles": len(tile_paths),
            "tile_size": self.tile_size,
            "stride": self.stride,
            "astroflow_version": self._astroflow_version(),
        }

        manifest_path = self.output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return manifest_path

    def validate_dataset(self) -> None:
        """Verify that the expected dataset artifacts exist."""
        errors: List[str] = []

        index_path = self.output_root / "index.csv"
        manifest_path = self.output_root / "manifest.json"

        tiles_exist = any(
            tile_dir.is_dir()
            for tile_dir in self.output_root.rglob("tiles")
        )

        if not tiles_exist:
            errors.append("Missing tiles directory under output root.")
        if not index_path.is_file():
            errors.append("Missing merged index.csv at output root.")
        if not manifest_path.is_file():
            errors.append("Missing manifest.json at output root.")

        if errors:
            raise RuntimeError("Dataset validation failed:\n" + "\n".join(errors))

        index = pd.read_csv(index_path)

        if "tile_filename" not in index.columns:
            errors.append("Expected 'tile_filename' column missing from index.csv.")
        else:
            duplicate_names = index["tile_filename"].duplicated()
            if duplicate_names.any():
                duplicates = index.loc[duplicate_names, "tile_filename"].unique()
                errors.append(
                    "Duplicate tile filenames found in index.csv: "
                    + ", ".join(duplicates)
                )

        tile_paths = list(self.output_root.rglob("*.npy"))
        tile_names = {path.name for path in tile_paths}

        if "tile_filename" in index.columns:
            missing_tiles = [
                filename
                for filename in index["tile_filename"].unique()
                if filename not in tile_names
            ]

            if missing_tiles:
                errors.append(
                    "Index refers to missing tiles: "
                    + ", ".join(missing_tiles[:10])
                )

        manifest = json.loads(manifest_path.read_text())
        expected_tile_count = manifest.get("num_tiles")

        if expected_tile_count is not None and len(tile_paths) != expected_tile_count:
            errors.append(
                f"Manifest num_tiles ({expected_tile_count}) does not match generated tiles ({len(tile_paths)})"
            )

        if errors:
            raise RuntimeError("Dataset validation failed:\n" + "\n".join(errors))

    def dataset_summary(self) -> None:
        """Print a dataset summary using the manifest and index artifacts."""
        manifest_path = self.output_root / "manifest.json"
        index_path = self.output_root / "index.csv"

        if not manifest_path.exists() or not index_path.exists():
            raise RuntimeError(
                "Cannot print dataset summary without manifest.json and index.csv."
            )

        manifest = json.loads(manifest_path.read_text())

        print("=" * 40)
        print("Dataset Summary")
        print("=" * 40)
        print(f"Input Images: {manifest.get('num_images', 'unknown')}")
        print(f"Generated Tiles: {manifest.get('num_tiles', 'unknown')}")
        print(f"Tile Size: {manifest.get('tile_size', self.tile_size)}")
        print(f"Stride: {manifest.get('stride', self.stride)}")
        print(f"Master Index: {index_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Output Directory: {self.output_root}")
        print("=" * 40)

    def build(self) -> None:
        """Construct the dataset from FITS files and finalize metadata."""
        fits_files = self.discover_fits()

        for fits_path in fits_files:
            self.process_observation(fits_path)

        self.merge_indices()
        self.build_manifest()
        self.validate_dataset()
        self.dataset_summary()

        tile_count = len(list(self.output_root.rglob("*.npy")))
        print("Dataset build complete")
        print(f"Input root: {self.input_root}")
        print(f"Output root: {self.output_root}")
        print(f"Images processed: {len(fits_files)}")
        print(f"Tiles generated: {tile_count}")
        print(f"Index file: {self.output_root / 'index.csv'}")
        print(f"Manifest file: {self.output_root / 'manifest.json'}")

    def _astroflow_version(self) -> str:
        """Resolve the Astro-Flow-3D package version if available."""
        try:
            return importlib.metadata.version("astro-flow-3d")
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0"
