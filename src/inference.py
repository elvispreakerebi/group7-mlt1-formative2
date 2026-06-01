from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.embeddings import encode_with_sentence_transformer
from src.evaluation import binarize_scores, multilabel_metrics, tune_label_thresholds
from src.experiments import _model_scores, make_split, word_char_logistic_pipeline
from src.preprocessing import prepare_text


def generate_final_predictions(bundle, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_embeddings = encode_with_sentence_transformer(
        bundle.train,
        model_name=config["experiments"]["sentence_transformer_model"],
        embedding_dir=config["paths"]["embedding_dir"],
        split_name="train",
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    test_embeddings = encode_with_sentence_transformer(
        bundle.test,
        model_name=config["experiments"]["sentence_transformer_model"],
        embedding_dir=config["paths"]["embedding_dir"],
        split_name="test",
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )

    train_text = prepare_text(
        bundle.train,
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )
    test_text = prepare_text(
        bundle.test,
        use_type_token=True,
        text_col=config["data"]["text_column"],
        type_col=config["data"]["type_column"],
    )

    split = make_split(
        train_text,
        bundle.labels,
        seed=config["seed"],
        validation_size=config["split"]["validation_size"],
    )
    tfidf_weight, thresholds, validation_metrics = tune_final_ensemble(
        train_text=train_text,
        train_embeddings=train_embeddings,
        labels=bundle.labels,
        split=split,
        threshold_grid=[float(value) for value in config["experiments"]["threshold_grid"]],
    )

    final_tfidf = word_char_logistic_pipeline(c_value=2.0, class_weight="balanced")
    final_tfidf.fit(train_text, bundle.labels.values)

    final_embedding_model = OneVsRestClassifier(
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    )
    final_embedding_model.fit(train_embeddings, bundle.labels.values)

    test_scores = (tfidf_weight * _model_scores(final_tfidf, test_text)) + (
        (1 - tfidf_weight) * _model_scores(final_embedding_model, test_embeddings)
    )
    predictions = binarize_scores(test_scores, threshold=thresholds)
    prediction_frame = pd.DataFrame(predictions, columns=bundle.label_names)
    prediction_frame.insert(0, config["data"]["id_column"], bundle.test[config["data"]["id_column"]].values)
    prediction_frame["predicted_labels"] = [
        "; ".join([label for label, value in zip(bundle.label_names, row) if value == 1])
        for row in predictions
    ]

    metadata = pd.DataFrame(
        [
            {
                "selected_model": "tfidf_minilm_ensemble",
                "tfidf_weight": tfidf_weight,
                "mean_threshold": float(np.mean(thresholds)),
                **validation_metrics,
            }
        ]
    )
    return prediction_frame, metadata


def tune_final_ensemble(
    train_text: pd.Series,
    train_embeddings: np.ndarray,
    labels: pd.DataFrame,
    split,
    threshold_grid: list[float],
) -> tuple[float, np.ndarray, dict[str, float]]:
    tfidf_model = word_char_logistic_pipeline(c_value=2.0, class_weight="balanced")
    tfidf_model.fit(split.x_train, split.y_train)
    tfidf_scores = _model_scores(tfidf_model, split.x_valid)

    embedding_model = OneVsRestClassifier(
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    )
    embedding_model.fit(train_embeddings[split.train_idx], split.y_train)
    embedding_scores = _model_scores(embedding_model, train_embeddings[split.valid_idx])

    best_weight = 0.5
    best_thresholds = np.full(labels.shape[1], 0.5)
    best_metrics: dict[str, float] | None = None
    for weight in [0.3, 0.5, 0.7]:
        scores = (weight * tfidf_scores) + ((1 - weight) * embedding_scores)
        thresholds = tune_label_thresholds(split.y_valid, scores, threshold_grid)
        predictions = binarize_scores(scores, thresholds)
        metrics = multilabel_metrics(split.y_valid, predictions)
        if best_metrics is None or metrics["hamming_loss"] < best_metrics["hamming_loss"]:
            best_weight = weight
            best_thresholds = thresholds
            best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_thresholds, best_metrics
