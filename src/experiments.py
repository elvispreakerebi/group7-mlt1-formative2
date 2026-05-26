from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import binarize_scores, label_wise_metrics, multilabel_metrics
from src.features import word_tfidf
from src.preprocessing import prepare_text


@dataclass(frozen=True)
class SplitData:
    x_train: pd.Series
    x_valid: pd.Series
    y_train: np.ndarray
    y_valid: np.ndarray


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
    scores = pipeline.predict_proba(split.x_valid)
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
