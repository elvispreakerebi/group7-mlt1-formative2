from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils import resolve_path


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    labels: pd.DataFrame
    label_names: list[str]
    mlb: MultiLabelBinarizer


def load_raw_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoding = config["data"]["encoding"]
    train = pd.read_csv(resolve_path(config["paths"]["train_csv"]), encoding=encoding)
    test = pd.read_csv(resolve_path(config["paths"]["test_csv"]), encoding=encoding)
    return train, test


def label_columns(frame: pd.DataFrame, prefix: str = "Label") -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def validate_schema(train: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> None:
    id_col = config["data"]["id_column"]
    type_col = config["data"]["type_column"]
    text_col = config["data"]["text_column"]
    required_common = {id_col, type_col, text_col}
    missing_train = required_common - set(train.columns)
    missing_test = required_common - set(test.columns)
    if missing_train:
        raise ValueError(f"Train data is missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test data is missing required columns: {sorted(missing_test)}")
    if not label_columns(train, config["data"]["label_prefix"]):
        raise ValueError("Train data does not contain any Label columns.")
    if train[text_col].isna().any() or test[text_col].isna().any():
        raise ValueError("Text column contains missing values; review source data before modeling.")


def parse_label_lists(train: pd.DataFrame, config: dict[str, Any]) -> list[list[str]]:
    labels = label_columns(train, config["data"]["label_prefix"])
    parsed: list[list[str]] = []
    for _, row in train[labels].iterrows():
        row_labels = [
            str(value).strip()
            for value in row.dropna().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        if not row_labels:
            raise ValueError("Every training row is expected to have at least one label.")
        parsed.append(row_labels)
    return parsed


def build_label_matrix(label_lists: list[list[str]]) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(label_lists)
    labels = pd.DataFrame(matrix, columns=mlb.classes_).astype("int16")
    return labels, mlb


def load_dataset(config: dict[str, Any]) -> DatasetBundle:
    train, test = load_raw_data(config)
    validate_schema(train, test, config)
    label_lists = parse_label_lists(train, config)
    labels, mlb = build_label_matrix(label_lists)
    return DatasetBundle(train=train, test=test, labels=labels, label_names=list(mlb.classes_), mlb=mlb)


def dataset_summary(bundle: DatasetBundle, config: dict[str, Any]) -> pd.DataFrame:
    text_col = config["data"]["text_column"]
    type_col = config["data"]["type_column"]
    labels_per_row = bundle.labels.sum(axis=1)
    summary = {
        "train_rows": len(bundle.train),
        "test_rows": len(bundle.test),
        "train_columns": len(bundle.train.columns),
        "test_columns": len(bundle.test.columns),
        "unique_labels": len(bundle.label_names),
        "missing_train_text": int(bundle.train[text_col].isna().sum()),
        "missing_test_text": int(bundle.test[text_col].isna().sum()),
        "unique_train_types": int(bundle.train[type_col].nunique()),
        "unique_test_types": int(bundle.test[type_col].nunique()),
        "mean_labels_per_row": round(float(labels_per_row.mean()), 4),
        "max_labels_per_row": int(labels_per_row.max()),
    }
    return pd.DataFrame([summary])


def save_table(frame: pd.DataFrame, path: str | Path) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, index=False)
