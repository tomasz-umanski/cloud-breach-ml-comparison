"""Generate all figures used in the report from the experiment results."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Use a project-local, writable cache dir to avoid font cache warnings.
_CACHE = Path(__file__).resolve().parents[1] / ".cache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .data.preprocess import Dataset  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})

# Stable colour per model for consistency across figures.
_COLORS = {
    "Logistic Regression": "#9e9e9e",
    "Random Forest": "#2e7d32",
    "MLP": "#1565c0",
    "CNN 1D": "#6a1b9a",
    "LSTM": "#e65100",
}
_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
_METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    out = config.FIGURES_DIR / name
    fig.savefig(out)
    # Mirror into the report so LaTeX can include it directly.
    shutil.copy(out, config.REPORT_FIGURES_DIR / name)
    plt.close(fig)
    return out


def plot_class_distribution(data: Dataset) -> None:
    names = data.class_names
    counts = np.bincount(
        np.concatenate([data.ym_train, data.ym_val, data.ym_test]),
        minlength=len(names),
    )
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([names[i] for i in order], counts[order], color="#1565c0")
    ax.set_ylabel("Liczba przepływów")
    ax.set_title("Rozkład klas w zbiorze danych")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(counts[order]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    _save(fig, "class_distribution.png")


def plot_metrics_comparison(results: dict) -> None:
    models = list(results.keys())
    x = np.arange(len(_METRIC_KEYS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        vals = [results[m]["metrics"][k] for k in _METRIC_KEYS]
        ax.bar(x + i * width, vals, width, label=m, color=_COLORS.get(m, None))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(_METRIC_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Porównanie metryk skuteczności modeli")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "metrics_comparison.png")


def plot_roc_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, r in results.items():
        c = r["metrics"]["roc_curve"]
        ax.plot(c["x"], c["y"], label=f"{m} (AUC={r['metrics']['roc_auc']:.3f})",
                color=_COLORS.get(m, None))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("Odsetek fałszywie pozytywnych (FPR)")
    ax.set_ylabel("Czułość (TPR)")
    ax.set_title("Krzywe ROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "roc_curves.png")


def plot_pr_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, r in results.items():
        c = r["metrics"]["pr_curve"]
        ax.plot(c["x"], c["y"], label=f"{m} (AP={r['metrics']['pr_auc']:.3f})",
                color=_COLORS.get(m, None))
    ax.set_xlabel("Czułość (Recall)")
    ax.set_ylabel("Precyzja (Precision)")
    ax.set_title("Krzywe precyzja-czułość")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    _save(fig, "pr_curves.png")


def plot_confusion_matrices(results: dict) -> None:
    models = list(results.keys())
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    # constrained_layout keeps subplots, titles and labels from overlapping.
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.6 * cols, 4.2 * rows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)
    for ax, m in zip(axes, models):
        cm = results[m]["metrics"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(m, fontsize=11, pad=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Atak"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Atak"], fontsize=9)
        ax.set_xlabel("Predykcja", fontsize=9)
        ax.set_ylabel("Rzeczywista", fontsize=9)
        thresh = mat.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black",
                        fontsize=11)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Macierze pomyłek (zbiór testowy)", fontsize=13)
    _save(fig, "confusion_matrices.png")


def plot_time_vs_performance(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, r in results.items():
        ax.scatter(r["train_time_s"], r["metrics"]["f1"], s=120,
                   color=_COLORS.get(m, None), edgecolor="black", zorder=3)
        ax.annotate(m, (r["train_time_s"], r["metrics"]["f1"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Czas treningu [s] (skala log)")
    ax.set_ylabel("F1 (zbiór testowy)")
    ax.set_title("Kompromis: koszt treningu a skuteczność")
    ax.grid(alpha=0.3)
    _save(fig, "time_vs_f1.png")


def plot_learning_curves(results: dict) -> None:
    deep = {m: r for m, r in results.items() if r.get("is_deep") and r.get("history")}
    if not deep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, r in deep.items():
        h = r["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=m, color=_COLORS.get(m, None))
        axes[1].plot(epochs, h["val_auc"], label=m, color=_COLORS.get(m, None))
    axes[0].set_title("Strata walidacyjna"); axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("BCE loss"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title("ROC-AUC walidacyjny"); axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("ROC-AUC"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.suptitle("Krzywe uczenia modeli głębokich")
    _save(fig, "learning_curves.png")


def plot_feature_importance(results: dict, top_k: int = 15) -> None:
    rf = results.get("Random Forest")
    if not rf or "feature_importances" not in rf:
        return
    items = sorted(rf["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color="#2e7d32")
    ax.set_xlabel("Ważność cechy (Gini importance)")
    ax.set_title(f"Najważniejsze cechy wg lasu losowego (top {top_k})")
    _save(fig, "feature_importance.png")


def generate_all(results: dict, data: Dataset) -> None:
    plot_class_distribution(data)
    plot_metrics_comparison(results)
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
    plot_time_vs_performance(results)
    plot_learning_curves(results)
    plot_feature_importance(results)
    print(f"[plots] Figures written to {config.FIGURES_DIR} and {config.REPORT_FIGURES_DIR}")
