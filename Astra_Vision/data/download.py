#!/usr/bin/env python3
"""
Download JWST data from MAST archive.

Supports multiple fields and bands:
- CEERS (Cosmic Evolution Early Release Science)
- COSMOS-Web
- JADES (JWST Advanced Deep Extragalactic Survey)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Mast, Observations


def download_ceers_data(output_dir: str | Path, bands: list[str] = None) -> dict:
    """
    Download CEERS JWST data.

    CEERS covers the ECDFS field with NIRCam and MIRI observations.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save downloaded data.
    bands : list[str], optional
        List of bands to download. Options: 'nircam', 'miri', or specific band names.
        If None, downloads all available bands.

    Returns
    -------
    dict
        Summary of downloaded data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CEERS field coordinates (ECDFS)
    # RA: 10:00:28.6, Dec: -04:48:42 (approximate center)
    ceers_coord = SkyCoord(
        ra=150.1187 * u.degree,
        dec=-4.8117 * u.degree,
        frame='icrs'
    )

    print("=" * 60)
    print("CEERS JWST Data Download")
    print("=" * 60)
    print(f"\nField: CEERS (ECDFS)")
    print(f"Coordinates: {ceers_coord.ra.deg:.4f}, {ceers_coord.dec.deg:.4f}")

    # Query MAST for CEERS observations
    print("\n[1/4] Querying MAST for observations...")

    # Filter by instrument
    instruments = []
    if bands is None or 'nircam' in bands:
        instruments.append('NIRCam')
    if bands is None or 'miri' in bands:
        instruments.append('MIRI')

    # Query with cone search using position and radius (obs_collection and instrument_name not supported by query_region)
    # Use 5 degrees radius to capture all JWST observations in the CEERS region
    obs_table = Observations.query_region(
        ceers_coord,
        radius=5 * u.degree
    )

    # Filter for JWST observations and by instrument after query
    if len(obs_table) > 0:
        # Filter for JWST obs_collection
        mask = np.array([str(row['obs_collection']) == 'JWST' for row in obs_table])
        obs_table = obs_table[mask]

        # Filter by instrument (handle numpy.str_ type)
        if instruments and len(obs_table) > 0:
            mask = np.array([any(inst in str(row['instrument_name']) for inst in instruments) for row in obs_table])
            obs_table = obs_table[mask]

    print(f"Found {len(obs_table)} observations")

    if len(obs_table) == 0:
        print("No observations found. Try expanding the search radius.")
        return {"downloaded": 0, "files": []}

    # Filter for actual imaging data (not spectroscopy)
    # Look for programs with 'image' in description or NIRCam/MIRI imaging filters
    print("\n[2/4] Filtering for imaging data...")

    # Get product list for each observation
    all_products = []
    for obs in obs_table:
        products = Observations.get_product_list(obs)
        # Manually filter for I2D files (productSubGroupDescription doesn't work)
        i2d_products = [
            p for p in products
            if 'i2d' in str(p['dataURI']).lower()
            and str(p['productType']) == 'SCIENCE'
        ]
        all_products.append({
            'observation_id': obs['obsid'],
            'products': i2d_products
        })

    # Download filtered products using REST API
    print("\n[3/4] Downloading data...")

    downloaded_files = []
    for item in all_products:
        products = item['products']
        if len(products) == 0:
            continue

        for product in products:
            try:
                # Use REST API to download (MAST download_products has issues)
                uri = product['dataURI']
                filename = product['productFilename']
                encoded_uri = requests.utils.quote(uri, safe='')
                download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={encoded_uri}"

                response = requests.get(download_url, timeout=120)
                if response.status_code == 200:
                    filepath = output_dir / filename
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded_files.append(str(filepath))
                else:
                    print(f"Warning: Could not download {uri}, status {response.status_code}")
            except Exception as e:
                print(f"Warning: Could not download {product['dataURI']}: {e}")

    print(f"\nDownloaded {len(downloaded_files)} files to {output_dir}")

    # Generate manifest
    manifest = {
        "field": "CEERS",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_observations": len(obs_table),
        "num_files": len(downloaded_files),
        "files": [str(Path(f).name) for f in downloaded_files],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")

    return {
        "downloaded": len(downloaded_files),
        "files": downloaded_files,
        "manifest": manifest
    }


def download_miri_deep_field(output_dir: str | Path) -> dict:
    """
    Download MIRI Deep Field data for NIRCam-dark galaxy search.

    This field has deep MIRI coverage with shallower NIRCam.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIRI Deep Field Download")
    print("=" * 60)

    # MIRI Deep Field coordinates
    mdf_coord = SkyCoord(
        ra=53.1625 * u.degree,  # COSMOS center
        dec=-27.7925 * u.degree,
        frame='icrs'
    )

    print(f"\nField: MIRI Deep Field (COSMOS)")
    print(f"Coordinates: {mdf_coord.ra.deg:.4f}, {mdf_coord.dec.deg:.4f}")

    # Query for observations (obs_collection not supported by query_region)
    obs_table = Observations.query_region(
        mdf_coord,
        radius=5 * u.arcmin
    )

    # Filter for JWST observations and MIRI instrument after query
    if len(obs_table) > 0:
        mask = np.array([str(row['obs_collection']) == 'JWST' for row in obs_table])
        obs_table = obs_table[mask]
        mask = np.array(['MIRI' in str(row['instrument_name']) for row in obs_table])
        obs_table = obs_table[mask]

    print(f"Found {len(obs_table)} MIRI observations")

    # Download using REST API
    downloaded_files = []
    for obs in obs_table:
        try:
            products = Observations.get_product_list(obs)
            # Manually filter for I2D files
            i2d_products = [
                p for p in products
                if 'i2d' in str(p['dataURI']).lower()
                and str(p['productType']) == 'SCIENCE'
            ]
            for product in i2d_products:
                try:
                    uri = product['dataURI']
                    filename = product['productFilename']
                    encoded_uri = requests.utils.quote(uri, safe='')
                    download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={encoded_uri}"

                    response = requests.get(download_url, timeout=120)
                    if response.status_code == 200:
                        filepath = output_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        downloaded_files.append(str(filepath))
                except Exception as e:
                    print(f"Warning: Could not download {uri}: {e}")
        except Exception as e:
            print(f"Error processing observation {obs['obsid']}: {e}")

    manifest = {
        "field": "MIRI_Deep_Field",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_observations": len(obs_table),
        "num_files": len(downloaded_files),
    }

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    return {
        "downloaded": len(downloaded_files),
        "files": downloaded_files
    }


def download_by_program_id(program_id: str, output_dir: str | Path) -> dict:
    """
    Download data by JWST program ID.

    Common public programs:
    - 1345: CEERS (NIRCam)
    - 1291: COSMOS-Web (NIRCam + MIRI)
    - 1423: JADES Deep (NIRCam)
    - 1288: JADES Deep (MIRI)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Program {program_id}...")

    obs_table = Observations.query_criteria(
        obs_id=f"*{program_id}*",
        obs_collection='JWST'
    )

    print(f"Found {len(obs_table)} observations")

    downloaded_files = []
    for obs in obs_table:
        try:
            products = Observations.get_product_list(obs)
            filtered = Observations.filter_products(
                products,
                productSubGroupDescription=['I2D']
            )
            for product in filtered:
                try:
                    result = Observations.download_product(
                        product['productUri'],
                        path=str(output_dir),
                        cache=True
                    )
                    if result is not None:
                        downloaded_files.append(result['Local Path'][0])
                except Exception as e:
                    print(f"Warning: {e}")
        except Exception as e:
            print(f"Error: {e}")

    return {"downloaded": len(downloaded_files), "files": downloaded_files}


def main():
    parser = argparse.ArgumentParser(
        description="Download JWST data from MAST archive"
    )
    parser.add_argument(
        "--field",
        type=str,
        choices=['ceers', 'mdf', 'cosmos', 'jades'],
        help="Predefined field to download"
    )
    parser.add_argument(
        "--program",
        type=str,
        help="Specific JWST program ID to download"
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="nircam,miri",
        help="Comma-separated list of bands (nircam, miri)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="downloads",
        help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.field:
        if args.field == 'ceers':
            result = download_ceers_data(output_dir / 'ceers', args.bands.split(','))
        elif args.field == 'mdf':
            result = download_miri_deep_field(output_dir / 'mdf')
        elif args.field == 'cosmos':
            result = download_ceers_data(output_dir / 'cosmos', args.bands.split(','))
        elif args.field == 'jades':
            result = download_by_program_id('1288', output_dir / 'jades')
    elif args.program:
        result = download_by_program_id(args.program, output_dir / f'program_{args.program}')
    else:
        print("Error: Must specify --field or --program")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Files downloaded: {result['downloaded']}")
    if 'manifest' in result:
        print(f"Manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
