"""Export LaTeX tables and a results summary so the report reflects real output."""
from __future__ import annotations

from pathlib import Path

from . import config
from .data.preprocess import Dataset

_MODEL_ORDER = ["Logistic Regression", "Random Forest", "MLP", "CNN 1D", "LSTM"]


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _best_per_column(results: dict, keys: list[str]) -> dict[str, str]:
    """Return, for each metric key, the model name with the highest value."""
    best = {}
    for k in keys:
        best[k] = max(results, key=lambda m: results[m]["metrics"][k])
    return best


def metrics_table(results: dict) -> str:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    header = "Model & Accuracy & Precision & Recall & F1 & ROC-AUC & PR-AUC \\\\"
    best = _best_per_column(results, keys)
    rows = []
    for m in _MODEL_ORDER:
        if m not in results:
            continue
        met = results[m]["metrics"]
        cells = []
        for k in keys:
            val = _fmt(met[k])
            if best[k] == m:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}"
    )


def efficiency_table(results: dict) -> str:
    header = ("Model & Czas treningu [s] & Czas inferencji [s] & "
              "Złożoność modelu \\\\")
    rows = []
    for m in _MODEL_ORDER:
        if m not in results:
            continue
        r = results[m]
        comp = r["complexity"]
        if "n_parameters" in comp:
            comp_str = f"{comp['n_parameters']:,} param."
        elif "total_tree_nodes" in comp:
            comp_str = f"{comp['n_estimators']} drzew / {comp['total_tree_nodes']:,} węzłów"
        else:
            comp_str = f"{comp.get('n_coefficients', 0)} wsp."
        comp_str = comp_str.replace(",", "\\,")
        rows.append(
            f"{m} & {_fmt(r['train_time_s'], 2)} & "
            f"{_fmt(r['inference_time_s'], 4)} & {comp_str} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrl}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}"
    )


def dataset_macros(data: Dataset, results: dict) -> str:
    """LaTeX \newcommand macros with key numbers, so prose stays in sync."""
    n_total = data.X_train.shape[0] + data.X_val.shape[0] + data.X_test.shape[0]
    best_f1_model = max(results, key=lambda m: results[m]["metrics"]["f1"])
    rf = results.get("Random Forest", {}).get("metrics", {})
    best_deep = max(
        (m for m in results if results[m]["is_deep"]),
        key=lambda m: results[m]["metrics"]["f1"],
        default="MLP",
    )
    macros = {
        "DataSource": data.source,
        "NTotal": f"{n_total:,}".replace(",", "\\,"),
        "NTrain": f"{data.X_train.shape[0]:,}".replace(",", "\\,"),
        "NVal": f"{data.X_val.shape[0]:,}".replace(",", "\\,"),
        "NTest": f"{data.X_test.shape[0]:,}".replace(",", "\\,"),
        "NFeatures": str(data.n_features),
        "NClasses": str(len(data.class_names)),
        "BestFOneModel": best_f1_model,
        "BestFOne": _fmt(results[best_f1_model]["metrics"]["f1"]),
        "RFFOne": _fmt(rf.get("f1", 0.0)),
        "RFAUC": _fmt(rf.get("roc_auc", 0.0)),
        "BestDeepModel": best_deep,
        "BestDeepFOne": _fmt(results[best_deep]["metrics"]["f1"]),
        "RFTrainTime": _fmt(results.get("Random Forest", {}).get("train_time_s", 0.0), 2),
        "BestDeepTrainTime": _fmt(results.get(best_deep, {}).get("train_time_s", 0.0), 2),
    }
    lines = [f"\\newcommand{{\\res{k}}}{{{v}}}" for k, v in macros.items()]
    return "\n".join(lines)


def export_all(results: dict, data: Dataset) -> None:
    out_dir: Path = config.PROJECT_ROOT / "report" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_table.tex").write_text(metrics_table(results), encoding="utf-8")
    (out_dir / "efficiency_table.tex").write_text(efficiency_table(results), encoding="utf-8")
    (out_dir / "macros.tex").write_text(dataset_macros(data, results), encoding="utf-8")
    print(f"[tables] LaTeX tables and macros written to {out_dir}")
