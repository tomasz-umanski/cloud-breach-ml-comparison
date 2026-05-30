"""Helper for obtaining a subset of the real CSE-CIC-IDS2018 dataset.

CSE-CIC-IDS2018 was collected by the Canadian Institute for Cybersecurity on an
AWS cloud testbed and contains benign traffic together with attacks such as
brute force, DoS/DDoS, web attacks, infiltration and botnet activity. The full
dataset is distributed as a set of large CSV files (several GB) and usually has
to be downloaded manually after accepting the licence.

This module does **not** try to scrape the data automatically (the official
distribution requires manual acceptance / an AWS CLI sync). Instead it looks for
already-downloaded CSV files in ``data/raw`` and loads/concatenates them. If no
files are present it raises ``DatasetUnavailable`` so the caller can fall back
to the synthetic generator.

To use the real data, place one or more CIC-IDS2018 CSV files in ``data/raw``,
for example::

    data/raw/Wednesday-14-02-2018.csv
    data/raw/Thursday-15-02-2018.csv

(see https://www.unb.ca/cic/datasets/ids-2018.html).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .. import config


class DatasetUnavailable(RuntimeError):
    """Raised when the real CIC-IDS2018 CSV files cannot be found."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip and snake_case the raw CIC column names."""
    df = df.rename(columns=lambda c: c.strip())
    # The dataset uses a 'Label' column; keep it under that exact name.
    rename = {}
    for col in df.columns:
        if col.lower() == "label":
            rename[col] = "Label"
    return df.rename(columns=rename)


def load_real_subset(
    max_rows: int = 120_000,
    chunksize: int = 200_000,
    sample_frac: float = 0.02,
) -> pd.DataFrame:
    """Load a memory-safe random subset of the CIC-IDS2018 CSVs in ``data/raw``.

    The full dataset is several GB (one daily file alone is ~3.8 GB), so the
    files are read in chunks and randomly down-sampled on the fly. A roughly
    balanced number of rows is drawn from each daily file (each day contains
    different attack types), and the result is capped at ``max_rows``.

    Parameters
    ----------
    max_rows:
        Total number of rows to keep across all files.
    chunksize:
        Number of rows read per chunk (controls peak memory use).
    sample_frac:
        Fraction of each chunk kept while streaming; the per-file buffer is
        additionally capped so memory stays bounded even for huge files.
    """
    raw_dir: Path = config.RAW_DIR
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise DatasetUnavailable(
            f"No CIC-IDS2018 CSV files found in {raw_dir}. "
            "Download them manually or use the synthetic generator."
        )

    # Cache the sampled subset so subsequent runs skip the multi-GB read.
    cache = config.PROCESSED_DIR / f"cicids_subset_{max_rows}.csv"
    if cache.exists():
        print(f"[download] Using cached subset: {cache}")
        return pd.read_csv(cache, low_memory=False)

    per_file = math.ceil(max_rows / len(csv_files))
    seed = config.SEED
    frames: list[pd.DataFrame] = []

    for path in csv_files:
        buffer: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = _normalise_columns(chunk)
            if 0.0 < sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=seed)
            if len(chunk):
                buffer.append(chunk)
            # Keep the per-file buffer bounded for very large files.
            if sum(len(b) for b in buffer) > per_file * 4:
                merged = pd.concat(buffer, ignore_index=True)
                buffer = [merged.sample(n=per_file * 2, random_state=seed)]

        if not buffer:
            continue
        file_df = pd.concat(buffer, ignore_index=True)
        if len(file_df) > per_file:
            file_df = file_df.sample(n=per_file, random_state=seed)
        frames.append(file_df)
        print(f"[download] {path.name}: kept {len(file_df):,} rows")

    df = pd.concat(frames, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.reset_index(drop=True)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"[download] Cached subset to {cache}")
    return df
