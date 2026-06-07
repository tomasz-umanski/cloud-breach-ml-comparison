"""Cleaning, encoding, scaling and splitting of the raw dataset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .. import config


@dataclass
class Dataset:
    """Container holding everything the models and plots need."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    # Binary targets: 0 = BENIGN, 1 = attack of any kind.
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    # Multi-class targets (attack family index), aligned with the splits above.
    ym_train: np.ndarray
    ym_val: np.ndarray
    ym_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]  # multi-class names, index == encoded value
    scaler: StandardScaler
    label_encoder: LabelEncoder
    source: str

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> dict:
        def counts(y: np.ndarray) -> dict[str, int]:
            uniq, cnt = np.unique(y, return_counts=True)
            return {int(u): int(c) for u, c in zip(uniq, cnt)}

        return {
            "source": self.source,
            "n_features": self.n_features,
            "n_train": int(self.X_train.shape[0]),
            "n_val": int(self.X_val.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "class_names": self.class_names,
            "binary_balance_train": counts(self.y_train),
            "binary_balance_test": counts(self.y_test),
        }


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric/identifier columns and handle NaN / Inf values."""
    df = df.copy()

    # Columns that are identifiers or non-informative in CIC-IDS exports.
    drop_candidates = [
        "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
        "Src Port", "Source Port", "Dst Port", "Destination Port",
        "Protocol", "Timestamp",
    ]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns],
                 errors="ignore")

    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the dataset.")

    # Some CIC-IDS exports contain repeated header rows as data; drop them.
    df = df[df["Label"].astype(str).str.strip().str.lower() != "label"]

    labels = df["Label"].astype(str).str.strip()
    features = df.drop(columns=["Label"])

    # Keep only numeric feature columns.
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.select_dtypes(include=[np.number])

    # Replace infinities and fill remaining NaNs with column medians.
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    # Drop constant columns (zero variance carries no information).
    nunique = features.nunique()
    features = features.loc[:, nunique[nunique > 1].index]

    cleaned = features.copy()
    cleaned["Label"] = labels.values
    return cleaned.dropna(subset=["Label"]).reset_index(drop=True)


def _binarize(labels: pd.Series) -> np.ndarray:
    """Map any non-benign label to 1, benign to 0."""
    benign = labels.str.upper().str.contains("BENIGN") | (labels.str.upper() == "NORMAL")
    return (~benign).astype(int).to_numpy()


def build_dataset(df: pd.DataFrame, cfg: config.ExperimentConfig, source: str) -> Dataset:
    """Full preprocessing pipeline: clean -> encode -> split -> scale."""
    cleaned = _clean(df)

    feature_names = [c for c in cleaned.columns if c != "Label"]
    X = cleaned[feature_names].to_numpy(dtype=np.float64)
    raw_labels = cleaned["Label"].astype(str).str.strip()

    # Multi-class encoding (attack family) and binary target.
    label_encoder = LabelEncoder()
    ym = label_encoder.fit_transform(raw_labels)
    y = _binarize(raw_labels)

    # Stratify on the binary target (benign vs. attack). Stratifying on the
    # multi-class label is fragile on the real dataset, where some attack
    # sub-types are extremely rare (a class may have a single sample).
    stratify = y if cfg.split.stratify else None
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )
    strat_train = y[train_idx] if cfg.split.stratify else None
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=cfg.split.val_size,
        random_state=cfg.seed,
        stratify=strat_train,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    return Dataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        ym_train=ym[train_idx],
        ym_val=ym[val_idx],
        ym_test=ym[test_idx],
        feature_names=feature_names,
        class_names=list(label_encoder.classes_),
        scaler=scaler,
        label_encoder=label_encoder,
        source=source,
    )
