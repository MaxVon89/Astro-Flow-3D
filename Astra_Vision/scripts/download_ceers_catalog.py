#!/usr/bin/env python3
"""
Download CEERS catalog data from MAST and match to tiles.

This script:
1. Queries MAST catalogs for CEERS field data (RA ~ 180, Dec ~ 62)
2. Matches catalog sources to tile positions
3. Extracts physical parameters: redshift, stellar mass, SFR, morphology
4. Saves results to targets.json and catalog_matches.json

Usage:
    python3 download_ceers_catalog.py [--force] [--cache-dir PATH]

The --force flag will re-download catalogs even if cached.
The --cache-dir specifies where to store cached catalog data.
"""
import json
import sys
import os
import hashlib
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.mast import Catalogs
import pickle


# CEERS field center (Bootes field)
CEERS_CENTER = SkyCoord(ra=180.0 * u.deg, dec=62.0 * u.deg, frame='icrs')
MATCH_RADIUS_ARCSEC = 2.0

# Catalogs to query
CATALOGS_TO_QUERY = ['3dhst', 'GLIMPSE', 'GSC', '2MASS', 'AllWISE', 'SDSS', 'VIKING', 'PS1']

# Output paths
OUTPUT_DIR = Path("/lp-dev/nvidia/projects/Astro-Flow-3D/Astra_Vision/data/catalog_matches")
CACHE_DIR = Path("/lp-dev/nvidia/projects/Astro-Flow-3D/.astro_cache")


def get_tile_manifests():
    """Load all tile manifest files from the outputs directory."""
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


def get_cache_path(catalog_name, center_ra, center_dec, radius_deg):
    """Get cache file path for a catalog query."""
    hash_input = f"{catalog_name}_{center_ra}_{center_dec}_{radius_deg}"
    hash_str = hashlib.md5(hash_input.encode()).hexdigest()
    return CACHE_DIR / f"{catalog_name}_{hash_str}.pkl"


def query_catalog(catalog_name, center_ra, center_dec, radius_deg, force=False):
    """
    Query a catalog from MAST, using cache if available.

    Returns catalog table or None if query fails.
    """
    cache_path = get_cache_path(catalog_name, center_ra, center_dec, radius_deg)

    # Try to load from cache
    if not force and cache_path.exists():
        print(f"  Loading {catalog_name} from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Query MAST
    try:
        print(f"  Querying {catalog_name}...")
        result = Catalogs.query_region(
            f"{center_ra} {center_dec}",
            radius=radius_deg * u.deg,
            catalog=catalog_name
        )

        if result is not None and len(result) > 0:
            print(f"    Found {len(result)} sources")

            # Cache the result
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"    Cached to {cache_path}")

            return result
        else:
            print(f"    No results")
            # Create empty cache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            empty_table = Table()
            with open(cache_path, 'wb') as f:
                pickle.dump(empty_table, f)
            return None

    except Exception as e:
        print(f"    Error: {e}")
        return None


def query_ceers_catalogs(center_ra, center_dec, radius_deg=1.0, force=False):
    """
    Query all MAST catalogs for CEERS field data.
    """
    print(f"\nQuerying MAST catalogs at RA={center_ra:.2f}, Dec={center_dec:.2f}")
    print(f"Radius: {radius_deg:.2f} deg")

    all_catalogs = {}

    for catalog_name in CATALOGS_TO_QUERY:
        catalog = query_catalog(catalog_name, center_ra, center_dec, radius_deg, force)
        if catalog is not None:
            all_catalogs[catalog_name] = catalog

    return all_catalogs


def match_catalog_to_tiles(catalogs, tile_manifests, match_radius_arcsec=MATCH_RADIUS_ARCSEC):
    """
    Match catalog sources to tile centers.
    """
    match_radius = match_radius_arcsec * u.arcsec

    # Combine all catalogs
    combined_catalog = None
    catalog_names = []

    for name, catalog in catalogs.items():
        if catalog is not None and len(catalog) > 0:
            catalog_names.append(name)
            if combined_catalog is None:
                combined_catalog = catalog
            else:
                try:
                    combined_catalog = vstack([combined_catalog, catalog], join_type='outer')
                except Exception:
                    pass

    print(f"\nCombined catalog: {len(catalog_names)} catalogs, {len(combined_catalog) if combined_catalog else 0} sources")

    # Extract RA/Dec columns
    ra_col = None
    dec_col = None

    if combined_catalog is not None:
        for col in ['ra', 'RA', 'RA_J2000', 'ra_j2000', 'ALPHA_J2000']:
            if col in combined_catalog.colnames:
                ra_col = col
                break

        for col in ['dec', 'DEC', 'Dec', 'dec_j2000', 'DELTA_J2000']:
            if col in combined_catalog.colnames:
                dec_col = col
                break

        if ra_col and dec_col:
            catalog_coords = SkyCoord(
                combined_catalog[ra_col] * u.deg,
                combined_catalog[dec_col] * u.deg,
                frame='icrs'
            )

    # Physical parameter column patterns
    redshift_cols = ['z', 'Z', 'redshift', 'Z_1', 'z_phot', 'Z_phot', 'z_best']
    mass_cols = ['logM', 'logMass', 'stellar_mass', 'MASS', 'M_*', 'logm']
    sfr_cols = ['SFR', 'sfr', 'logSFR', 'SFR_IR', 'SFR_UV', 'SFR_tot']
    morph_cols = ['morph', 'MORPH', 'class', 'CLASS']

    results = {
        'ceers_field': {
            'center_ra': center_ra,
            'center_dec': center_dec,
            'radius_arcmin': radius_deg * 60
        },
        'tiles': {},
        'catalog_info': {
            'catalogs_queried': catalog_names,
            'total_sources': len(combined_catalog) if combined_catalog else 0,
            'ra_column': ra_col,
            'dec_column': dec_col
        }
    }

    targets = {
        'tiles': [],
        'catalog_parameters': [],
        'center': {'ra': center_ra, 'dec': center_dec},
        'metadata': {
            'match_radius_arcsec': match_radius_arcsec,
            'catalogs_queried': catalog_names,
            'field': 'CEERS',
            'program': 'jw02733'
        }
    }

    if combined_catalog is not None:
        targets['catalog_parameters'] = list(combined_catalog.colnames)[:20]

    print("\nMatching tiles to catalog sources...")

    tile_count = 0
    match_count = 0

    for tile_set_name, manifest in tile_manifests.items():
        for tile_info in manifest.get('tiles', []):
            tile_name = Path(tile_info['path']).stem

            # Calculate tile center approximate RA/Dec
            tile_size = tile_info.get('tile_size', 256)
            x_center = tile_info['x'] + tile_size / 2
            y_center = tile_info['y'] + tile_size / 2

            # Approximate conversion (1 deg = 3600 arcsec)
            tile_ra = center_ra + (x_center - 1024) / 3600.0
            tile_dec = center_dec + (y_center - 1024) / 3600.0

            tile_result = {
                'tile_id': tile_name,
                'tile_set': tile_set_name,
                'path': tile_info['path'],
                'position': {'x': tile_info['x'], 'y': tile_info['y']},
                'tile_size': tile_size,
                'center_ra': tile_ra,
                'center_dec': tile_dec
            }

            # Find matches if we have catalog data
            if combined_catalog is not None and ra_col and dec_col:
                tile_coord = SkyCoord(ra=tile_ra * u.deg, dec=tile_dec * u.deg, frame='icrs')
                idx, sep, _ = tile_coord.search_around_sky(catalog_coords, match_radius)

                if len(idx) > 0:
                    matched_sources = combined_catalog[idx]
                    tile_result['catalog_matches'] = {
                        'count': len(idx),
                        'sources': []
                    }

                    match_count += len(idx)

                    for source in matched_sources:
                        source_data = {
                            'ra': float(source[ra_col]),
                            'dec': float(source[dec_col]),
                            'catalog': catalog_names[0] if catalog_names else 'combined'
                        }

                        # Extract physical parameters
                        for col in redshift_cols:
                            if col in source.colnames:
                                try:
                                    source_data['redshift'] = float(source[col])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        for col in mass_cols:
                            if col in source.colnames:
                                try:
                                    source_data['stellar_mass'] = float(source[col])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        for col in sfr_cols:
                            if col in source.colnames:
                                try:
                                    source_data['sfr'] = float(source[col])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        for col in morph_cols:
                            if col in source.colnames:
                                try:
                                    source_data['morphology'] = str(source[col])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        tile_result['catalog_matches']['sources'].append(source_data)

                        if not 'best_match' in tile_result:
                            tile_result['best_match'] = {
                                'redshift': source_data.get('redshift'),
                                'stellar_mass': source_data.get('stellar_mass'),
                                'sfr': source_data.get('sfr'),
                                'morphology': source_data.get('morphology')
                            }

            results['tiles'][tile_name] = tile_result

            physical_params = tile_result.get('best_match', {
                'redshift': None, 'stellar_mass': None, 'sfr': None, 'morphology': None
            })

            targets['tiles'].append({
                'tile_id': tile_name,
                'path': tile_result['path'],
                'position': tile_result['position'],
                'center_ra': tile_ra,
                'center_dec': tile_dec,
                'catalog_matches_count': tile_result.get('catalog_matches', {}).get('count', 0),
                'physical_parameters': physical_params
            })

            tile_count += 1

    print(f"  Processed {tile_count} tiles, found {match_count} catalog matches")

    return results, targets


def save_results(results, targets, output_dir):
    """Save results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog_file = output_path / "catalog_matches.json"
    targets_file = output_path / "targets.json"

    with open(catalog_file, 'w') as f:
        json.dump(results, f, indent=2)

    with open(targets_file, 'w') as f:
        json.dump(targets, f, indent=2)

    print(f"\nSaved results to:")
    print(f"  {catalog_file}")
    print(f"  {targets_file}")

    return catalog_file, targets_file


def main():
    """Main function."""
    print("=" * 60)
    print("CEERS Catalog Download and Tile Matching")
    print("=" * 60)

    # Parse command line arguments
    force_download = '--force' in sys.argv
    cache_dir = CACHE_DIR

    # Output directory
    output_dir = OUTPUT_DIR

    # Load tile manifests
    print("\n1. Loading tile manifests...")
    tile_manifests = get_tile_manifests()

    # Query catalogs
    print("\n2. Querying MAST catalogs...")
    catalogs = query_ceers_catalogs(
        CEERS_CENTER.ra.deg,
        CEERS_CENTER.dec.deg,
        radius_deg=0.5,
        force=force_download
    )

    # Match to tiles
    print("\n3. Matching catalogs to tiles...")
    results, targets = match_catalog_to_tiles(catalogs, tile_manifests, match_radius_arcsec=MATCH_RADIUS_ARCSEC)

    # Save results
    print("\n4. Saving results...")
    catalog_file, targets_file = save_results(results, targets, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Tiles processed: {len(results['tiles'])}")
    print(f"  Catalogs queried: {len(catalogs)}")
    print(f"  Total catalog sources: {results['catalog_info'].get('total_sources', 0)}")
    print(f"  Total matches: {sum(t.get('catalog_matches_count', 0) for t in targets['tiles'])}")

    print("\nDone!")
    print("=" * 60)

    return catalog_file, targets_file


if __name__ == "__main__":
    main()
