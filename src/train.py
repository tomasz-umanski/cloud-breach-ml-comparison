"""Train every model on the same split and collect metrics + timings."""
from __future__ import annotations

import numpy as np

from . import config, evaluate
from .data.preprocess import Dataset
from .models.classical import LogisticRegressionModel, RandomForestModel
from .models.deep import DeepModel
from .utils import Timer


def build_models(cfg: config.ExperimentConfig, n_features: int) -> list:
    """Instantiate all models to be compared."""
    return [
        LogisticRegressionModel(seed=cfg.seed),
        RandomForestModel(cfg.random_forest, seed=cfg.seed),
        DeepModel("MLP", n_features, cfg.deep, seed=cfg.seed),
        DeepModel("CNN1D", n_features, cfg.deep, seed=cfg.seed),
        DeepModel("LSTM", n_features, cfg.deep, seed=cfg.seed),
    ]


def train_and_evaluate(data: Dataset, cfg: config.ExperimentConfig) -> dict:
    """Train all models and return a results dict keyed by model name."""
    results: dict = {}
    models = build_models(cfg, data.n_features)

    for model in models:
        print(f"\n=== Training: {model.name} ===")
        with Timer() as t_train:
            model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
        train_time = t_train.elapsed

        y_prob = model.predict_proba(data.X_test)
        metrics = evaluate.compute_metrics(data.y_test, y_prob)
        infer_time = evaluate.measure_inference_time(model, data.X_test)

        results[model.name] = {
            "is_deep": bool(model.is_deep),
            "metrics": metrics,
            "train_time_s": float(train_time),
            "inference_time_s": float(infer_time),
            "inference_per_1k_ms": float(infer_time / max(len(data.X_test), 1) * 1e6),
            "complexity": model.complexity(),
            "history": getattr(model, "history", None),
        }
        print(
            f"    F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
            f"train={train_time:.2f}s  infer={infer_time:.3f}s"
        )

        # Keep feature importances from the random forest for the feature-importance plot.
        if isinstance(model, RandomForestModel):
            results[model.name]["feature_importances"] = {
                name: float(imp)
                for name, imp in zip(data.feature_names, model.feature_importances())
            }

    return results
