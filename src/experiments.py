from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation import binarize_scores, label_wise_metrics, multilabel_metrics
from src.evaluation import tune_label_thresholds
from src.features import combined_word_char_tfidf, word_tfidf
from src.preprocessing import prepare_text


@dataclass(frozen=True)
class SplitData:
    x_train: pd.Series
    x_valid: pd.Series
    y_train: np.ndarray
    y_valid: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray


def make_split(text: pd.Series, labels: pd.DataFrame, seed: int, validation_size: float) -> SplitData:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
        train_idx, valid_idx = next(splitter.split(text, labels.values))
    except ModuleNotFoundError:
        train_idx, valid_idx = train_test_split(
            np.arange(len(text)),
            test_size=validation_size,
            random_state=seed,
            shuffle=True,
        )
    return SplitData(
        x_train=text.iloc[train_idx],
        x_valid=text.iloc[valid_idx],
        y_train=labels.iloc[train_idx].values,
        y_valid=labels.iloc[valid_idx].values,
        train_idx=train_idx,
        valid_idx=valid_idx,
    )


def logistic_pipeline(
    ngram_range: tuple[int, int],
    max_features: int | None,
    min_df: int,
    max_df: float,
    c_value: float,
    class_weight: str | None = None,
) -> Pipeline:
    classifier = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    return Pipeline(
        [
            ("tfidf", word_tfidf(ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df)),
            ("model", OneVsRestClassifier(classifier)),
        ]
    )


def linear_svm_pipeline(
    ngram_range: tuple[int, int],
    max_features: int | None,
    min_df: int,
    max_df: float,
    c_value: float,
) -> Pipeline:
    classifier = LinearSVC(C=c_value, class_weight="balanced", random_state=42, max_iter=5000)
    return Pipeline(
        [
            ("tfidf", word_tfidf(ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df)),
            ("model", OneVsRestClassifier(classifier)),
        ]
    )


def word_char_logistic_pipeline(c_value: float, class_weight: str | None = "balanced") -> Pipeline:
    classifier = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    return Pipeline(
        [
            ("tfidf", combined_word_char_tfidf()),
            ("model", OneVsRestClassifier(classifier)),
        ]
    )


def run_pipeline_experiment(
    experiment_id: str,
    rationale: str,
    insight: str,
    train_frame: pd.DataFrame,
    labels: pd.DataFrame,
    label_names: list[str],
    config: dict[str, Any],
    use_type_token: bool,
    pipeline: Pipeline,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    text = prepare_text(
        train_frame,
        use_type_token=use_type_token,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    split = make_split(text, labels, seed=config["seed"], validation_size=config["split"]["validation_size"])
    start = time.perf_counter()
    pipeline.fit(split.x_train, split.y_train)
    runtime_seconds = time.perf_counter() - start
    scores = _model_scores(pipeline, split.x_valid)
    predictions = binarize_scores(scores, threshold=threshold)
    metrics = multilabel_metrics(split.y_valid, predictions)
    result = {
        "experiment_id": experiment_id,
        "rationale": rationale,
        "threshold": threshold,
        "runtime_seconds": round(runtime_seconds, 4),
        "interpretation": insight,
        **metrics,
    }
    return result, label_wise_metrics(split.y_valid, predictions, label_names, experiment_id), pipeline


def run_embedding_experiment(
    experiment_id: str,
    rationale: str,
    insight: str,
    embeddings: np.ndarray,
    labels: pd.DataFrame,
    label_names: list[str],
    config: dict[str, Any],
    model_kind: str,
    threshold: float = 0.5,
    c_value: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    split = _split_arrays(embeddings, labels, seed=config["seed"], validation_size=config["split"]["validation_size"])
    start = time.perf_counter()
    if model_kind == "ovr_logreg":
        model = OneVsRestClassifier(
            LogisticRegression(C=c_value, solver="liblinear", max_iter=1000, random_state=42)
        )
    elif model_kind == "classifier_chain":
        label_order = np.argsort(-labels.sum(axis=0).to_numpy())
        model = ClassifierChain(
            LogisticRegression(C=c_value, solver="liblinear", max_iter=1000, random_state=42),
            order=label_order,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported embedding model_kind: {model_kind}")
    model.fit(split["x_train"], split["y_train"])
    runtime_seconds = time.perf_counter() - start
    scores = _model_scores(model, split["x_valid"])
    predictions = binarize_scores(scores, threshold=threshold)
    metrics = multilabel_metrics(split["y_valid"], predictions)
    result = {
        "experiment_id": experiment_id,
        "rationale": rationale,
        "threshold": threshold,
        "runtime_seconds": round(runtime_seconds, 4),
        "interpretation": insight,
        **metrics,
    }
    return result, label_wise_metrics(split["y_valid"], predictions, label_names, experiment_id), model


def run_threshold_tuned_pipeline_experiment(
    experiment_id: str,
    rationale: str,
    insight: str,
    train_frame: pd.DataFrame,
    labels: pd.DataFrame,
    label_names: list[str],
    config: dict[str, Any],
    use_type_token: bool,
    pipeline: Pipeline,
    threshold_grid: list[float],
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline, np.ndarray]:
    text = prepare_text(
        train_frame,
        use_type_token=use_type_token,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    split = make_split(text, labels, seed=config["seed"], validation_size=config["split"]["validation_size"])
    start = time.perf_counter()
    pipeline.fit(split.x_train, split.y_train)
    runtime_seconds = time.perf_counter() - start
    scores = _model_scores(pipeline, split.x_valid)
    thresholds = tune_label_thresholds(split.y_valid, scores, threshold_grid)
    predictions = binarize_scores(scores, threshold=thresholds)
    metrics = multilabel_metrics(split.y_valid, predictions)
    result = {
        "experiment_id": experiment_id,
        "rationale": rationale,
        "threshold": "per-label tuned",
        "runtime_seconds": round(runtime_seconds, 4),
        "interpretation": insight,
        "mean_threshold": round(float(thresholds.mean()), 4),
        **metrics,
    }
    return result, label_wise_metrics(split.y_valid, predictions, label_names, experiment_id), pipeline, thresholds


def run_ensemble_experiment(
    experiment_id: str,
    rationale: str,
    insight: str,
    train_frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: pd.DataFrame,
    label_names: list[str],
    config: dict[str, Any],
    tfidf_pipeline: Pipeline,
    weights: list[float],
    threshold_grid: list[float],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    text = prepare_text(
        train_frame,
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    split = make_split(text, labels, seed=config["seed"], validation_size=config["split"]["validation_size"])
    start = time.perf_counter()
    tfidf_pipeline.fit(split.x_train, split.y_train)
    tfidf_scores = _model_scores(tfidf_pipeline, split.x_valid)

    embedding_model = OneVsRestClassifier(
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    )
    embedding_model.fit(embeddings[split.train_idx], split.y_train)
    embedding_scores = _model_scores(embedding_model, embeddings[split.valid_idx])

    best: dict[str, Any] | None = None
    for weight in weights:
        ensemble_scores = (weight * tfidf_scores) + ((1 - weight) * embedding_scores)
        thresholds = tune_label_thresholds(split.y_valid, ensemble_scores, threshold_grid)
        predictions = binarize_scores(ensemble_scores, threshold=thresholds)
        metrics = multilabel_metrics(split.y_valid, predictions)
        if best is None or metrics["hamming_loss"] < best["metrics"]["hamming_loss"]:
            best = {
                "weight": weight,
                "thresholds": thresholds,
                "predictions": predictions,
                "metrics": metrics,
            }
    runtime_seconds = time.perf_counter() - start
    assert best is not None
    result = {
        "experiment_id": experiment_id,
        "rationale": rationale,
        "threshold": "per-label tuned",
        "runtime_seconds": round(runtime_seconds, 4),
        "interpretation": insight,
        "ensemble_tfidf_weight": best["weight"],
        "mean_threshold": round(float(best["thresholds"].mean()), 4),
        **best["metrics"],
    }
    artifacts = {
        "tfidf_pipeline": tfidf_pipeline,
        "embedding_model": embedding_model,
        "thresholds": best["thresholds"],
        "tfidf_weight": best["weight"],
    }
    return result, label_wise_metrics(split.y_valid, best["predictions"], label_names, experiment_id), artifacts


def _model_scores(pipeline: Pipeline, text: pd.Series) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(text)
    scores = pipeline.decision_function(text)
    return np.asarray(scores)


def _split_arrays(features: np.ndarray, labels: pd.DataFrame, seed: int, validation_size: float) -> dict[str, np.ndarray]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
        train_idx, valid_idx = next(splitter.split(features, labels.values))
    except ModuleNotFoundError:
        train_idx, valid_idx = train_test_split(
            np.arange(len(features)),
            test_size=validation_size,
            random_state=seed,
            shuffle=True,
        )
    return {
        "x_train": features[train_idx],
        "x_valid": features[valid_idx],
        "y_train": labels.iloc[train_idx].values,
        "y_valid": labels.iloc[valid_idx].values,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
    }
