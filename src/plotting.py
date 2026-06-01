from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, resolve_path


def save_barplot(frame: pd.DataFrame, x: str, y: str, title: str, path: str | Path, rotation: int = 45) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = resolve_path(path)
    ensure_dir(output.parent)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=frame, x=x, y=y, color="#3366AA")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_histogram(series: pd.Series, title: str, xlabel: str, path: str | Path, bins: int = 30) -> None:
    import matplotlib.pyplot as plt

    output = resolve_path(path)
    ensure_dir(output.parent)
    plt.figure(figsize=(10, 6))
    plt.hist(series, bins=bins, color="#3366AA", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_heatmap(matrix: pd.DataFrame, title: str, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = resolve_path(path)
    ensure_dir(output.parent)
    plt.figure(figsize=(14, 11))
    sns.heatmap(matrix, cmap="Blues", square=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_metric_barplot(
    results: pd.DataFrame,
    metric: str,
    title: str,
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = resolve_path(path)
    ensure_dir(output.parent)
    ordered = results.sort_values(metric)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ordered, x="experiment_id", y=metric, color="#3366AA")
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_label_error_heatmap(label_metrics: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = resolve_path(path)
    ensure_dir(output.parent)
    pivot = label_metrics.pivot_table(index="label", columns="experiment_id", values="f1", aggfunc="mean")
    top_labels = label_metrics.groupby("label")["support"].max().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(13, 9))
    sns.heatmap(pivot.loc[top_labels], cmap="YlGnBu", annot=False)
    plt.title("Label-wise F1 for Top 15 Labels")
    plt.xlabel("Experiment")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
