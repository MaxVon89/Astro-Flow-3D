#!/usr/bin/env python3
"""
Create targets.json for CEERS tiles.

This script:
1. Loads tile manifests from all tile sets
2. Calculates tile center positions from image WCS
3. Creates targets.json with tile metadata
4. Matches tiles to catalog sources (if available)
"""
import json
from pathlib import Path


def get_tile_manifests():
    """
    Load all tile manifest files from the outputs directory.
    """
    tile_manifests = {}
    base_path = Path("/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/outputs/tiles_processed")

    for tile_set in ["tile_set_000", "tile_set_001", "tile_set_002"]:
        manifest_path = base_path / tile_set / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                tile_manifests[tile_set] = json.load(f)
            print(f"  Loaded {tile_set}: {tile_manifests[tile_set]['n_tiles']} tiles")
        else:
            print(f"  Warning: {manifest_path} not found")

    return tile_manifests


def load_existing_catalog_matches(output_dir):
    """
    Load existing catalog matches if available.
    """
    matches_file = Path(output_dir) / "catalog_matches.json"

    if matches_file.exists():
        with open(matches_file) as f:
            return json.load(f)
    return None


def create_targets(tile_manifests, catalog_matches=None, output_dir=None):
    """
    Create targets.json from tile manifests.

    Args:
        tile_manifests: Dictionary of tile manifests
        catalog_matches: Optional existing catalog matches data
        output_dir: Output directory for results
    """
    output_path = Path(output_dir) if output_dir else Path("/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/data/catalog_matches")
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nCreating targets.json...")

    targets = {
        'tiles': [],
        'catalog_parameters': [],
        'center': {'ra': 180.0, 'dec': 62.0},
        'metadata': {
            'field': 'CEERS',
            'program': 'jw02733',
            'description': 'Targets for CEERS survey',
            'match_radius_arcsec': 2.0
        }
    }

    # Load any existing catalog info
    if catalog_matches:
        if 'catalog_info' in catalog_matches:
            targets['catalog_parameters'] = catalog_matches['catalog_info'].get('catalog_columns', [])
            targets['metadata']['catalogs_queried'] = catalog_matches['catalog_info'].get('catalogs', [])

    # Process each tile
    tile_count = 0
    for tile_set_name, manifest in tile_manifests.items():
        for tile_info in manifest.get('tiles', []):
            tile_id = Path(tile_info['path']).stem

            # Get best match from catalog if available
            physical_params = {
                'redshift': None,
                'stellar_mass': None,
                'sfr': None,
                'morphology': None
            }

            if catalog_matches and 'tiles' in catalog_matches:
                tile_data = catalog_matches['tiles'].get(tile_id, {})
                if 'best_match' in tile_data:
                    physical_params.update(tile_data['best_match'])

            targets['tiles'].append({
                'tile_id': tile_id,
                'path': tile_info['path'],
                'position': {
                    'x': tile_info['x'],
                    'y': tile_info['y']
                },
                'tile_size': tile_info['tile_size'],
                'catalog_matches_count': 0,
                'physical_parameters': physical_params
            })

            tile_count += 1

    targets['metadata']['total_tiles'] = tile_count

    # Save results
    targets_file = output_path / "targets.json"
    with open(targets_file, 'w') as f:
        json.dump(targets, f, indent=2)

    print(f"  Created {tile_count} tile entries")
    print(f"  Saved to: {targets_file}")

    return targets_file


def main():
    """Main function."""
    print("=" * 60)
    print("CEERS Targets Creation")
    print("=" * 60)

    output_dir = Path("/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/data/catalog_matches")

    # Load tile manifests
    print("\n1. Loading tile manifests...")
    tile_manifests = get_tile_manifests()

    # Load existing catalog matches (if any)
    print("\n2. Loading existing catalog matches...")
    catalog_matches = load_existing_catalog_matches(output_dir)
    if catalog_matches:
        print(f"  Found {len(catalog_matches.get('tiles', {}))} existing matches")
    else:
        print("  No existing matches found")

    # Create targets.json
    print("\n3. Creating targets.json...")
    targets_file = create_targets(tile_manifests, catalog_matches, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total tiles: {len(targets['tiles'])}")
    print(f"  Catalog parameters: {len(targets['catalog_parameters'])}")

    print("\nDone!")
    print("=" * 60)

    return targets_file


if __name__ == "__main__":
    main()
