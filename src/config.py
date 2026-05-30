"""Central configuration for the experiment pipeline.

All paths are resolved relative to the project root so the pipeline can be run
from any working directory. Tweak the values here to change dataset size,
training length or the data source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

# --- Reproducibility --------------------------------------------------------
SEED = 42

# --- Data source ------------------------------------------------------------
# "auto"      -> try the real CSE-CIC-IDS2018 subset, fall back to synthetic
# "synthetic" -> always use the generated synthetic dataset (offline, fast)
# "cicids"    -> require the real dataset (error out if unavailable)
DATA_SOURCE = "auto"


@dataclass
class SyntheticConfig:
    """Parameters of the synthetic cloud network-flow generator."""

    n_samples: int = 60_000
    n_features: int = 30
    # Approximate share of benign traffic; the rest is split across attacks.
    benign_ratio: float = 0.70
    # Attack families modelled by the generator (besides BENIGN).
    attack_types: tuple[str, ...] = (
        "DoS",
        "BruteForce",
        "WebAttack",
        "Infiltration",
        "Botnet",
    )
    noise: float = 0.85


@dataclass
class SplitConfig:
    test_size: float = 0.20
    val_size: float = 0.20  # fraction of the *train* part used for validation
    stratify: bool = True


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str | None = "balanced"
    n_jobs: int = -1


@dataclass
class DeepConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    dropout: float = 0.3
    early_stopping_patience: int = 6


@dataclass
class ExperimentConfig:
    data_source: str = DATA_SOURCE
    seed: int = SEED
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)


def ensure_dirs() -> None:
    """Create all output directories if they do not exist yet."""
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORT_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
