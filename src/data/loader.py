"""Resolve the configured data source into a raw labelled DataFrame."""
from __future__ import annotations

import pandas as pd

from .. import config
from . import synthetic
from .download import DatasetUnavailable, load_real_subset


def load_raw(cfg: config.ExperimentConfig) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source_name)`` according to ``cfg.data_source``.

    * ``synthetic`` -> always generate synthetic data
    * ``cicids``    -> require the real dataset (raise if unavailable)
    * ``auto``      -> try the real dataset, fall back to synthetic
    """
    source = cfg.data_source

    if source == "synthetic":
        return _synthetic(cfg), "synthetic"

    if source == "cicids":
        return load_real_subset(), "cicids2018"

    # auto
    try:
        df = load_real_subset()
        return df, "cicids2018"
    except DatasetUnavailable as exc:
        print(f"[loader] Real dataset unavailable ({exc}). Using synthetic data.")
        return _synthetic(cfg), "synthetic"


def _synthetic(cfg: config.ExperimentConfig) -> pd.DataFrame:
    s = cfg.synthetic
    return synthetic.generate(
        n_samples=s.n_samples,
        n_features=s.n_features,
        benign_ratio=s.benign_ratio,
        attack_types=s.attack_types,
        noise=s.noise,
        seed=cfg.seed,
    )
