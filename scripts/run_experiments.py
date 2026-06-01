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
    run_ensemble_experiment,
    run_pipeline_experiment,
    run_threshold_tuned_pipeline_experiment,
    word_char_logistic_pipeline,
)
from src.plotting import save_label_error_heatmap, save_metric_barplot
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

    threshold_grid = [float(value) for value in config["experiments"]["threshold_grid"]]
    tuned_result, tuned_label_metrics, _, _ = run_threshold_tuned_pipeline_experiment(
        experiment_id="exp07_tuned_word_char_logreg",
        rationale="Tunes per-label thresholds for the strongest probability-based TF-IDF model.",
        insight="This directly optimizes the assignment metric because Hamming Loss depends on binary label decisions.",
        train_frame=bundle.train,
        labels=bundle.labels,
        label_names=bundle.label_names,
        config=config,
        use_type_token=True,
        pipeline=word_char_logistic_pipeline(c_value=2.0, class_weight="balanced"),
        threshold_grid=threshold_grid,
    )
    rows.append(tuned_result)
    label_rows.append(tuned_label_metrics)
    print(f"{tuned_result['experiment_id']}: hamming_loss={tuned_result['hamming_loss']:.4f}, micro_f1={tuned_result['micro_f1']:.4f}")

    ensemble_result, ensemble_label_metrics, _ = run_ensemble_experiment(
        experiment_id="exp08_tfidf_minilm_ensemble",
        rationale="Combines the best sparse TF-IDF signal with MiniLM semantic embeddings.",
        insight="This tests whether lexical phrase matching and semantic similarity make complementary errors.",
        train_frame=bundle.train,
        embeddings=embeddings,
        labels=bundle.labels,
        label_names=bundle.label_names,
        config=config,
        tfidf_pipeline=word_char_logistic_pipeline(c_value=2.0, class_weight="balanced"),
        weights=[0.3, 0.5, 0.7],
        threshold_grid=threshold_grid,
    )
    rows.append(ensemble_result)
    label_rows.append(ensemble_label_metrics)
    print(f"{ensemble_result['experiment_id']}: hamming_loss={ensemble_result['hamming_loss']:.4f}, micro_f1={ensemble_result['micro_f1']:.4f}")

    results = pd.DataFrame(rows)
    label_metrics = pd.concat(label_rows, ignore_index=True)
    final_comparison = results.sort_values("hamming_loss").reset_index(drop=True)
    save_table(results, f"{tables_dir}/experiment_results.csv")
    save_table(label_metrics, f"{tables_dir}/label_wise_metrics.csv")
    save_table(final_comparison, f"{tables_dir}/final_model_comparison.csv")
    save_metric_barplot(
        final_comparison,
        metric="hamming_loss",
        title="Validation Hamming Loss by Experiment",
        path=f"{config['paths']['figures_dir']}/experiment_hamming_loss.png",
    )
    save_metric_barplot(
        final_comparison.sort_values("micro_f1", ascending=False),
        metric="micro_f1",
        title="Validation Micro F1 by Experiment",
        path=f"{config['paths']['figures_dir']}/f1_comparison.png",
    )
    save_label_error_heatmap(label_metrics, path=f"{config['paths']['figures_dir']}/label_error_heatmap.png")
    print(f"Experiment outputs written to {resolve_path(tables_dir)}")


if __name__ == "__main__":
    main()
