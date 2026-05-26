from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import load_dataset, save_table
from src.experiments import logistic_pipeline, run_pipeline_experiment
from src.utils import ensure_output_dirs, load_config, resolve_path


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    bundle = load_dataset(config)
    tables_dir = config["paths"]["tables_dir"]

    experiments = [
        {
            "experiment_id": "exp01_tfidf_logreg_baseline",
            "rationale": "Baseline sparse lexical model using word TF-IDF and one-vs-rest Logistic Regression.",
            "insight": "This sets the first Hamming Loss benchmark using exact-word and phrase evidence.",
            "use_type_token": False,
            "pipeline": logistic_pipeline((1, 2), 20_000, 1, 1.0, 1.0),
            "threshold": 0.5,
        },
        {
            "experiment_id": "exp02_preprocessed_type_logreg",
            "rationale": "Adds text cleanup, a source Type token, stricter document frequency limits, and longer word n-grams.",
            "insight": "This tests whether source metadata and cleaner phrase features improve the baseline.",
            "use_type_token": True,
            "pipeline": logistic_pipeline((1, 3), 20_000, 2, 0.95, 1.0),
            "threshold": 0.5,
        },
    ]

    rows = []
    label_rows = []
    for experiment in experiments:
        result, label_metrics, _ = run_pipeline_experiment(
            experiment_id=experiment["experiment_id"],
            rationale=experiment["rationale"],
            insight=experiment["insight"],
            train_frame=bundle.train,
            labels=bundle.labels,
            label_names=bundle.label_names,
            config=config,
            use_type_token=experiment["use_type_token"],
            pipeline=experiment["pipeline"],
            threshold=experiment["threshold"],
        )
        rows.append(result)
        label_rows.append(label_metrics)
        print(f"{result['experiment_id']}: hamming_loss={result['hamming_loss']:.4f}, micro_f1={result['micro_f1']:.4f}")

    save_table(pd.DataFrame(rows), f"{tables_dir}/experiment_results.csv")
    save_table(pd.concat(label_rows, ignore_index=True), f"{tables_dir}/label_wise_metrics.csv")
    print(f"Experiment outputs written to {resolve_path(tables_dir)}")


if __name__ == "__main__":
    main()
