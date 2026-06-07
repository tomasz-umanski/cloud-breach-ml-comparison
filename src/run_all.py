"""End-to-end pipeline: load data -> preprocess -> train -> evaluate -> figures.

Run with::

    python -m src.run_all                 # uses config.DATA_SOURCE (default: auto)
    python -m src.run_all --source synthetic
    python -m src.run_all --epochs 10 --n-samples 20000   # quick smoke run
"""
from __future__ import annotations

import argparse

from . import config, plots
from .data.loader import load_raw
from .data.preprocess import build_dataset
from .train import train_and_evaluate
from .utils import save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data-breach forecasting experiment")
    p.add_argument("--source", choices=["auto", "synthetic", "cicids"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> config.ExperimentConfig:
    cfg = config.ExperimentConfig()
    if args.source:
        cfg.data_source = args.source
    if args.epochs:
        cfg.deep.epochs = args.epochs
    if args.n_samples:
        cfg.synthetic.n_samples = args.n_samples
    if args.seed:
        cfg.seed = args.seed
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    config.ensure_dirs()
    set_global_seed(cfg.seed)

    print(f"[run] data_source={cfg.data_source} seed={cfg.seed}")
    raw_df, source = load_raw(cfg)
    print(f"[run] loaded {len(raw_df):,} rows from source='{source}'")

    data = build_dataset(raw_df, cfg, source)
    summary = data.summary()
    print(f"[run] dataset summary: {summary}")
    save_json(summary, config.METRICS_DIR / "dataset_summary.json")

    results = train_and_evaluate(data, cfg)

    # Persist metrics (drop bulky curve arrays for the compact summary file).
    compact = {
        m: {
            "is_deep": r["is_deep"],
            "metrics": {k: v for k, v in r["metrics"].items()
                        if k not in ("roc_curve", "pr_curve")},
            "train_time_s": r["train_time_s"],
            "inference_time_s": r["inference_time_s"],
            "complexity": r["complexity"],
        }
        for m, r in results.items()
    }
    save_json(compact, config.METRICS_DIR / "results_summary.json")
    save_json(results, config.METRICS_DIR / "results_full.json")

    plots.generate_all(results, data)
    print("\n[run] Done. See results/.")


if __name__ == "__main__":
    main()
