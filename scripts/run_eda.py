from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import dataset_summary, load_dataset, save_table
from src.plotting import save_barplot, save_heatmap, save_histogram
from src.preprocessing import prepare_text
from src.utils import ensure_output_dirs, load_config, resolve_path


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    bundle = load_dataset(config)
    tables_dir = config["paths"]["tables_dir"]
    figures_dir = config["paths"]["figures_dir"]
    type_col = config["data"]["type_column"]
    text_col = config["data"]["text_column"]

    summary = dataset_summary(bundle, config)
    save_table(summary, f"{tables_dir}/dataset_summary.csv")

    label_frequency = (
        bundle.labels.sum(axis=0)
        .sort_values(ascending=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    label_frequency["share_of_rows"] = label_frequency["count"] / len(bundle.train)
    save_table(label_frequency, f"{tables_dir}/label_frequency.csv")

    label_cardinality = bundle.labels.sum(axis=1).value_counts().sort_index().rename_axis("labels_per_row").reset_index(name="rows")
    save_table(label_cardinality, f"{tables_dir}/label_cardinality.csv")

    type_distribution = bundle.train[type_col].value_counts().rename_axis("type").reset_index(name="count")
    save_table(type_distribution, f"{tables_dir}/type_distribution.csv")

    cleaned = prepare_text(bundle.train, use_type_token=False, text_col=text_col, type_col=type_col)
    text_lengths = cleaned.str.split().map(len)
    text_length_summary = text_lengths.describe().reset_index()
    text_length_summary.columns = ["statistic", "word_count"]
    save_table(text_length_summary, f"{tables_dir}/text_length_summary.csv")

    cooccurrence = pd.DataFrame(bundle.labels.T.dot(bundle.labels), index=bundle.label_names, columns=bundle.label_names)
    save_table(cooccurrence.reset_index(names="label"), f"{tables_dir}/label_cooccurrence.csv")

    save_barplot(
        label_frequency.head(15),
        x="label",
        y="count",
        title="Top 15 SDG 3 Indicator Labels",
        path=f"{figures_dir}/label_frequency.png",
    )
    save_barplot(
        label_cardinality,
        x="labels_per_row",
        y="rows",
        title="Number of Labels Assigned per Training Row",
        path=f"{figures_dir}/label_cardinality.png",
        rotation=0,
    )
    save_barplot(
        type_distribution,
        x="type",
        y="count",
        title="Training Samples by Source Type",
        path=f"{figures_dir}/type_distribution.png",
    )
    save_histogram(
        text_lengths,
        title="Document Length Distribution After Cleaning",
        xlabel="Words per document",
        path=f"{figures_dir}/text_length_distribution.png",
        bins=40,
    )
    top_labels = label_frequency.head(15)["label"].tolist()
    save_heatmap(
        cooccurrence.loc[top_labels, top_labels],
        title="Label Co-occurrence Among Top 15 Labels",
        path=f"{figures_dir}/label_cooccurrence_heatmap.png",
    )
    print(f"EDA outputs written to {resolve_path(tables_dir)} and {resolve_path(figures_dir)}")


if __name__ == "__main__":
    main()
