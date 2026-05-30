"""Synthetic cloud network-flow generator.

The generator produces a tabular dataset whose columns mimic the flow-based
features of the CSE-CIC-IDS2018 dataset (durations, packet/byte counts, rates,
TCP flag counts, window sizes ...). It models one benign class and several
attack families. Class distributions overlap and contain deliberately
non-linear interactions, so that:

* a linear baseline (logistic regression) is clearly outperformed,
* tree ensembles (random forest) and neural networks both reach high but not
  trivial accuracy, leaving room for a meaningful comparison.

This keeps the whole pipeline fully reproducible and runnable offline when the
real CSE-CIC-IDS2018 download is not available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd",
    "total_len_bwd",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "flow_bytes_s",
    "flow_pkts_s",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "fwd_psh_flags",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "urg_flag_cnt",
    "down_up_ratio",
    "avg_pkt_size",
    "init_win_fwd",
    "init_win_bwd",
    "active_mean",
    "idle_mean",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
)


def _class_profiles(n_features: int, rng: np.random.Generator) -> dict[str, dict]:
    """Build a mean/scale profile for every traffic class.

    Each profile is a Gaussian centre plus per-feature scale. Centres are
    placed at moderate, partially overlapping distances so the problem stays
    non-trivial.
    """
    classes = ["BENIGN", "DoS", "BruteForce", "WebAttack", "Infiltration", "Botnet"]
    profiles: dict[str, dict] = {}
    for label in classes:
        if label == "BENIGN":
            center = rng.normal(0.0, 0.4, size=n_features)
        else:
            # Attacks live further from the origin, each in its own direction.
            direction = rng.normal(0.0, 1.0, size=n_features)
            direction /= np.linalg.norm(direction) + 1e-9
            distance = rng.uniform(1.4, 2.6)
            center = direction * distance + rng.normal(0.0, 0.3, size=n_features)
        scale = rng.uniform(0.5, 1.3, size=n_features)
        profiles[label] = {"center": center, "scale": scale}
    return profiles


def _to_realistic_units(latent: np.ndarray) -> np.ndarray:
    """Map standardized latent features to plausible positive flow magnitudes."""
    # Exponential mapping keeps everything positive and heavy-tailed, like real
    # network-flow statistics, while preserving the latent class structure.
    return np.expm1(np.clip(latent, -6, 6)) + np.abs(latent)


def generate(
    n_samples: int,
    n_features: int,
    benign_ratio: float,
    attack_types: tuple[str, ...],
    noise: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a synthetic flow dataset as a labelled DataFrame.

    Returns a DataFrame with ``n_features`` numeric columns plus a ``Label``
    column holding the multi-class attack family (``BENIGN`` for normal flows).
    """
    rng = np.random.default_rng(seed)
    n_features = min(n_features, len(FEATURE_NAMES))
    feature_names = list(FEATURE_NAMES[:n_features])

    profiles = _class_profiles(n_features, rng)

    # Compose the label vector with the requested class balance.
    n_benign = int(round(n_samples * benign_ratio))
    n_attack_total = n_samples - n_benign
    # Slightly uneven attack split to mimic real-world imbalance.
    weights = rng.uniform(0.6, 1.4, size=len(attack_types))
    weights /= weights.sum()
    attack_counts = np.floor(weights * n_attack_total).astype(int)
    attack_counts[-1] += n_attack_total - attack_counts.sum()

    labels: list[str] = ["BENIGN"] * n_benign
    for atype, count in zip(attack_types, attack_counts):
        labels.extend([atype] * int(count))
    labels = np.array(labels, dtype=object)
    rng.shuffle(labels)

    latent = np.empty((n_samples, n_features), dtype=np.float64)
    for label in profiles:
        mask = labels == label
        k = int(mask.sum())
        if k == 0:
            continue
        prof = profiles[label]
        latent[mask] = (
            prof["center"]
            + rng.normal(0.0, 1.0, size=(k, n_features)) * prof["scale"] * noise
        )

    # Inject non-linear feature interactions (XOR-like and multiplicative) so
    # that linear models cannot capture the full structure.
    if n_features >= 6:
        latent[:, 3] += 1.5 * np.sign(latent[:, 0] * latent[:, 1])
        latent[:, 4] += latent[:, 2] ** 2 * 0.4
        latent[:, 5] += np.sin(latent[:, 1] * 2.0) * 1.2

    features = _to_realistic_units(latent)

    df = pd.DataFrame(features, columns=feature_names)
    # A few count-like features should be integers.
    for col in ("total_fwd_packets", "total_bwd_packets", "syn_flag_cnt",
                "fin_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt"):
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    df["Label"] = labels
    return df.reset_index(drop=True)
