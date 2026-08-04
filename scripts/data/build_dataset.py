from pathlib import Path

import yaml

from src.data.dataset_builder import DatasetBuilder


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml"
    config = load_config(config_path)

    builder = DatasetBuilder(
        input_root=config["paths"]["input_root"],
        output_root=config["paths"]["output_root"],
        tile_size=config["tiling"]["tile_size"],
        stride=config["tiling"]["stride"],
    )
    builder.build()


if __name__ == "__main__":
    main()
