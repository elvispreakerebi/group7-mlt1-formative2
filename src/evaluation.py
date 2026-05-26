from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score


def binarize_scores(scores: np.ndarray, threshold: float | np.ndarray = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def label_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], experiment_id: str) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(label_names):
        rows.append(
            {
                "experiment_id": experiment_id,
                "label": label,
                "support": int(y_true[:, index].sum()),
                "f1": float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "precision": float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "recall": float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            }
        )
    return pd.DataFrame(rows)
