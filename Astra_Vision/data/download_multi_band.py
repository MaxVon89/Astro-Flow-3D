#!/usr/bin/env python3
"""
Download multi-band JWST data (6 NIRCam + 4 MIRI bands) from MAST.

Targets CEERS and COSMOS-Web fields with full band coverage.
"""

import argparse
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import requests

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Mast, Observations


def get_nircam_bands():
    """Return NIRCam band filters (6 bands for training)."""
    return [
        'F115W', 'F150W', 'F200W',  # NIRCam short wavelength
        'F277W', 'F356W', 'F444W',  # NIRCam long wavelength
    ]


def get_miri_bands():
    """Return MIRI band filters (4 bands for training)."""
    return [
        'F770W', 'F1000W',  # MIRI short wavelength
        'F1130W', 'F1500W',  # MIRI long wavelength
    ]


def get_all_ceers_programs():
    """Return all CEERS-related JWST program IDs."""
    # CEERS is primarily program 2733 (OASIS), with related programs
    return ['2733', '1423', '1345', '2107', '1419', '1436', '1459', '1517', '1579']


def get_all_cosmos_programs():
    """Return all COSMOS-related JWST program IDs."""
    # COSMOS-Web is program 2107
    return ['2107', '1424', '1425', '1426', '1427', '1428', '1429', '1430']


def download_ceers_multiband(output_dir: str | Path, bands: list[str] = None) -> dict:
    """
    Download CEERS data with full NIRCam + MIRI band coverage.

    CEERS program: 2733
    Field: ECDFS (Extended Chandra Deep Field South)

    Parameters
    ----------
    output_dir : str | Path
        Directory to save downloaded data.
    bands : list[str], optional
        Specific bands to download. If None, downloads all 10 bands.

    Returns
    -------
    dict
        Summary of downloaded data with file paths and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CEERS field coordinates (ECDFS)
    ceers_coord = SkyCoord(
        ra=150.1187 * u.degree,
        dec=-4.8117 * u.degree,
        frame='icrs'
    )

    print("=" * 60)
    print("CEERS Multi-Band JWST Data Download")
    print("=" * 60)
    print(f"\nTarget: CEERS field (ECDFS)")
    print(f"Coordinates: RA={ceers_coord.ra.deg:.4f}, Dec={ceers_coord.dec.deg:.4f}")

    # Default to all 10 bands if not specified
    if bands is None:
        bands = get_nircam_bands() + get_miri_bands()

    print(f"\nRequesting {len(bands)} bands:")
    for b in bands:
        print(f"  - {b}")

    all_files = []
    file_info = []

    # Query MAST for CEERS observations using different approaches
    print("\nQuerying MAST for JWST observations...")

    try:
        # Query by position and instrument
        print("\n[1/3] Querying by position and instrument...")

        # Use a larger search area
        obs_table = Observations.query_region(
            ceers_coord,
            radius=10 * u.degree,
            instruments=['NIRCam', 'MIRI'],
            obs_collection='JWST'
        )

        print(f"Found {len(obs_table)} observations in CEERS region")

        if len(obs_table) == 0:
            print("No observations found. Trying different query...")
            # Alternative: query all JWST observations and filter
            obs_table = Observations.query_criteria(
                s_ra=(145, 155),
                s_dec=(-10, 0),
                obs_collection='JWST',
            )
            print(f"Found {len(obs_table)} observations in wider search")

        # Get products for each observation
        print("\n[2/3] Getting product lists...")

        all_products = []
        for i, obs in enumerate(obs_table):
            if i > 20:  # Limit to 20 observations for initial download
                break
            try:
                products = Observations.get_product_list(obs)
                # Filter for i2d files (final calibrated mosaics)
                i2d_products = [
                    p for p in products
                    if 'i2d' in str(p['dataURI']).lower()
                    and str(p['productType']) == 'SCIENCE'
                ]
                if len(i2d_products) > 0:
                    all_products.append({
                        'observation_id': obs['obsid'],
                        'products': i2d_products,
                        'filters': obs.get('filters', '')
                    })
            except Exception as e:
                print(f"  Warning: Could not get products for obs {obs['obsid']}: {e}")

        print(f"Found {len(all_products)} observations with i2d products")

        # Download filtered products
        print("\n[3/3] Downloading data...")

        downloaded_count = 0
        for item in all_products:
            products = item['products']
            if len(products) == 0:
                continue

            for product in products:
                try:
                    uri = product['dataURI']
                    filename = product['productFilename']
                    encoded_uri = requests.utils.quote(uri, safe='')
                    download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={encoded_uri}"

                    # Check if any target band is in the filename
                    band_found = None
                    for band in bands:
                        if band.lower() in filename.lower():
                            band_found = band
                            break

                    if band_found:
                        response = requests.get(download_url, timeout=120)
                        if response.status_code == 200:
                            filepath = output_dir / filename
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            all_files.append(str(filepath))
                            file_info.append({
                                'filename': filename,
                                'band': band_found,
                                'obsid': item['observation_id'],
                            })
                            downloaded_count += 1
                            print(f"  [{downloaded_count}] Downloaded: {filename} ({band_found})")
                        else:
                            print(f"  Warning: Could not download {uri}, status {response.status_code}")
                    else:
                        # Check if we should still download based on program/observation
                        # (sometimes filenames don't have band info)
                        obs_id = item['observation_id']
                        # Download all i2d files for CEERS programs
                        if any(str(p) in obs_id for p in get_all_ceers_programs()):
                            response = requests.get(download_url, timeout=120)
                            if response.status_code == 200:
                                filepath = output_dir / filename
                                with open(filepath, 'wb') as f:
                                    f.write(response.content)
                                all_files.append(str(filepath))
                                file_info.append({
                                    'filename': filename,
                                    'band': 'unknown',
                                    'obsid': item['observation_id'],
                                })
                                downloaded_count += 1
                                print(f"  [{downloaded_count}] Downloaded: {filename} (CEERS program)")
                except Exception as e:
                    print(f"  Warning: Could not download {product['dataURI']}: {e}")

    except Exception as e:
        print(f"Error during download: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal files downloaded: {len(all_files)}")

    # Generate manifest
    summary = {
        "field": "CEERS",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_bands_requested": len(bands),
        "bands": bands,
        "num_files": len(all_files),
        "files": file_info,
    }

    # Count files per band
    band_counts = {}
    for info in file_info:
        band = info['band']
        if band != 'unknown':
            band_counts[band] = band_counts.get(band, 0) + 1

    summary["band_counts"] = band_counts

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total files: {len(all_files)}")
    print("Per-band counts:")
    for band, count in sorted(band_counts.items()):
        status = "OK" if count > 0 else "MISSING"
        print(f"  {band}: {count} files [{status}]")

    return summary


def download_cosmos_web_multiband(output_dir: str | Path, bands: list[str] = None) -> dict:
    """
    Download COSMOS-Web data with NIRCam + MIRI coverage.

    COSMOS-Web program: 2107
    Field: COSMOS field
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # COSMOS coordinates
    cosmos_coord = SkyCoord(
        ra=150.1 * u.degree,
        dec=2.2 * u.degree,
        frame='icrs'
    )

    print("=" * 60)
    print("COSMOS-Web Multi-Band JWST Data Download")
    print("=" * 60)

    if bands is None:
        bands = get_nircam_bands() + get_miri_bands()

    print(f"Requesting {len(bands)} bands: {', '.join(bands)}")

    all_files = []
    file_info = []

    try:
        print(f"\nQuerying MAST for COSMOS-Web observations...")

        obs_table = Observations.query_region(
            cosmos_coord,
            radius=5 * u.degree,
            instruments=['NIRCam', 'MIRI'],
            obs_collection='JWST'
        )

        print(f"Found {len(obs_table)} observations")

        # Get products and download
        for obs in obs_table:
            try:
                products = Observations.get_product_list(obs)

                i2d_products = [
                    p for p in products
                    if 'i2d' in str(p['dataURI']).lower()
                    and str(p['productType']) == 'SCIENCE'
                ]

                for product in i2d_products:
                    data_uri = str(product['dataURI'])
                    filename = str(product['productFilename'])

                    for band in bands:
                        if band.lower() in filename.lower() or band.lower() in data_uri.lower():
                            try:
                                encoded_uri = requests.utils.quote(data_uri, safe='')
                                download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={encoded_uri}"

                                print(f"  Downloading {filename} ({band})...")
                                response = requests.get(download_url, timeout=180)

                                if response.status_code == 200:
                                    filepath = output_dir / filename
                                    with open(filepath, 'wb') as f:
                                        f.write(response.content)
                                    all_files.append(str(filepath))
                                    file_info.append({
                                        'filename': filename,
                                        'band': band,
                                        'obsid': obs['obsid']
                                    })
                                    print(f"    -> Saved: {filepath}")
                                break
                            except Exception as e:
                                print(f"    Error: {e}")
                            break
            except Exception as e:
                print(f"  Error: {e}")

    except Exception as e:
        print(f"Query error: {e}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            "field": "COSMOS-Web",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_bands": len(bands),
            "files": len(all_files),
        }, f, indent=2)

    return {"downloaded": len(all_files), "files": all_files}


def download_ceers_by_program(output_dir: str | Path, program_id: str = "2733") -> dict:
    """
    Download CEERS data by specific program ID.
    Program 2733 is the main CEERS program.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Program {program_id} from MAST...")

    try:
        # Query by program ID
        obs_table = Observations.query_criteria(
            obs_id=f"*{program_id}*",
            obs_collection='JWST',
        )

        print(f"Found {len(obs_table)} observations for program {program_id}")

        if len(obs_table) == 0:
            # Try alternative query format
            obs_table = Observations.query_criteria(
                program_id=program_id,
                obs_collection='JWST',
            )
            print(f"Found {len(obs_table)} observations (alternative query)")

        all_files = []
        file_info = []

        # Get products for each observation
        for obs in obs_table:
            try:
                products = Observations.get_product_list(obs)

                # Filter for i2d products
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
                            local_path = result.get('Local Path')
                            if local_path:
                                all_files.append(str(local_path))
                                file_info.append({
                                    'filename': Path(local_path).name,
                                    'band': 'unknown',
                                    'obsid': obs['obsid'],
                                })
                                print(f"  Downloaded: {Path(local_path).name}")
                    except Exception as e:
                        print(f"  Warning: Could not download {product['productUri']}: {e}")
            except Exception as e:
                print(f"  Error processing observation {obs['obsid']}: {e}")

        print(f"\nTotal files downloaded: {len(all_files)}")

        return {"downloaded": len(all_files), "files": all_files, "file_info": file_info}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"downloaded": 0, "files": []}


def main():
    parser = argparse.ArgumentParser(
        description="Download multi-band JWST data (6 NIRCam + 4 MIRI)"
    )
    parser.add_argument(
        "--field",
        type=str,
        choices=['ceers', 'cosmos'],
        default='ceers',
        help="Field to download (default: ceers)"
    )
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        help="Specific JWST program ID to download"
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help="Comma-separated specific bands (e.g., 'F115W,F150W,F200W')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="downloads",
        help="Output directory (default: downloads)"
    )
    parser.add_argument(
        "--by-program",
        action="store_true",
        help="Download by program ID instead of region query"
    )

    args = parser.parse_args()

    # Parse bands if specified
    bands = None
    if args.bands:
        bands = [b.strip() for b in args.bands.split(',')]

    output_dir = Path(args.output)

    if args.program:
        # Download by program ID
        program_output = output_dir / f"program_{args.program}"
        result = download_ceers_by_program(program_output, args.program)
    elif args.by_program:
        # Use program-based download for more reliable results
        output_dir = output_dir / 'ceers'
        result = download_ceers_by_program(output_dir, "2733")
    elif args.field == 'ceers':
        result = download_ceers_multiband(output_dir / 'ceers', bands)
    else:
        result = download_cosmos_web_multiband(output_dir / 'cosmos', bands)

    print(f"\nDownload complete. Files in: {output_dir}")

    if result.get('downloaded', 0) == 0:
        print("\nWARNING: No files were downloaded!")
        print("Common issues:")
        print("  1. MAST API rate limiting - wait and retry")
        print("  2. Program ID may have changed - check MAST website")
        print("  3. Network connectivity issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
