"""
IBTrACS Best-Track Downloader.

IBTrACS (International Best Track Archive for Climate Stewardship, NOAA NCEI) is the
authoritative ground-truth record of historical tropical cyclones: position, maximum
sustained wind, central pressure and radius of maximum winds at 3-hourly/6-hourly
intervals for every recorded storm.

This module fetches the two basin files relevant to this project:
  - NI (North Indian)  -> Bay of Bengal + Arabian Sea
  - SI (South Indian)  -> South Indian Ocean

Downloaded CSVs land in data/raw/best_track/ and are consumed by
src/data_pipeline/cyclone_dataset.py.

Usage:
    python -m src.data_pipeline.download_ibtracs
    python -m src.data_pipeline.download_ibtracs --basins NI SI --since 1980
"""

import argparse
import logging
import os
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

IBTRACS_BASE = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv"
)

# IBTrACS basin codes relevant to the Indian Ocean domain
BASIN_FILES: Dict[str, str] = {
    "NI": "ibtracs.NI.list.v04r01.csv",  # North Indian: BOB + ARB
    "SI": "ibtracs.SI.list.v04r01.csv",  # South Indian
}

DEFAULT_OUT_DIR = os.path.join("data", "raw", "best_track")


def download_basin(basin: str, out_dir: str = DEFAULT_OUT_DIR, timeout: int = 120) -> str:
    """
    Downloads a single IBTrACS basin CSV. Returns the local path.
    """
    if basin not in BASIN_FILES:
        raise ValueError(f"Unknown IBTrACS basin '{basin}'. Expected one of {list(BASIN_FILES)}")

    os.makedirs(out_dir, exist_ok=True)
    filename = BASIN_FILES[basin]
    url = f"{IBTRACS_BASE}/{filename}"
    dest = os.path.join(out_dir, filename)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info(f"[{basin}] already present, skipping download: {dest}")
        return dest

    logger.info(f"[{basin}] downloading {url}")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        tmp = dest + ".part"
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
        os.replace(tmp, dest)

    logger.info(f"[{basin}] saved {written / 1e6:.1f} MB -> {dest}")
    return dest


def download_all(basins: List[str] = None, out_dir: str = DEFAULT_OUT_DIR) -> Dict[str, str]:
    """
    Downloads every requested basin file. Returns {basin_code: local_path}.
    """
    basins = basins or list(BASIN_FILES.keys())
    paths = {}
    for b in basins:
        paths[b] = download_basin(b, out_dir=out_dir)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IBTrACS best-track CSVs.")
    parser.add_argument("--basins", nargs="+", default=["NI", "SI"], choices=list(BASIN_FILES))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = download_all(args.basins, args.out_dir)

    print("\nIBTrACS download complete:")
    for basin, path in paths.items():
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [{basin}] {path}  ({size_mb:.1f} MB)")
    print("\nNext: python -m src.data_pipeline.cyclone_dataset --inspect")


if __name__ == "__main__":
    main()
