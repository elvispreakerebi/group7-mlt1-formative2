from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import load_dataset, save_table
from src.embeddings import encode_with_sentence_transformer
from src.experiments import (
    linear_svm_pipeline,
    logistic_pipeline,
    run_embedding_experiment,
    run_pipeline_experiment,
    word_char_logistic_pipeline,
)
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
        {
            "experiment_id": "exp03_tfidf_linear_svm",
            "rationale": "Uses a margin-based Linear SVM with balanced class weights on word TF-IDF features.",
            "insight": "This tests whether a large-margin classifier handles sparse SDG indicator language better than Logistic Regression.",
            "use_type_token": True,
            "pipeline": linear_svm_pipeline((1, 2), 20_000, 2, 0.95, 1.0),
            "threshold": 0.0,
        },
        {
            "experiment_id": "exp04_word_char_balanced_logreg",
            "rationale": "Combines word and character TF-IDF features with class-balanced Logistic Regression.",
            "insight": "Character n-grams should help with acronyms, noisy wording, and rare health indicator phrasing.",
            "use_type_token": True,
            "pipeline": word_char_logistic_pipeline(c_value=2.0, class_weight="balanced"),
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

    embeddings = encode_with_sentence_transformer(
        bundle.train,
        model_name=config["experiments"]["sentence_transformer_model"],
        embedding_dir=config["paths"]["embedding_dir"],
        split_name="train",
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    embedding_experiments = [
        {
            "experiment_id": "exp05_minilm_logreg",
            "rationale": "Uses pretrained MiniLM sentence embeddings with one-vs-rest Logistic Regression.",
            "insight": "This tests semantic representation against sparse lexical TF-IDF features.",
            "model_kind": "ovr_logreg",
            "threshold": 0.5,
        },
        {
            "experiment_id": "exp06_minilm_classifier_chain",
            "rationale": "Uses MiniLM sentence embeddings with a Classifier Chain ordered by label frequency.",
            "insight": "This tests whether modeling label dependencies improves multi-label predictions.",
            "model_kind": "classifier_chain",
            "threshold": 0.5,
        },
    ]
    for experiment in embedding_experiments:
        result, label_metrics, _ = run_embedding_experiment(
            experiment_id=experiment["experiment_id"],
            rationale=experiment["rationale"],
            insight=experiment["insight"],
            embeddings=embeddings,
            labels=bundle.labels,
            label_names=bundle.label_names,
            config=config,
            model_kind=experiment["model_kind"],
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
