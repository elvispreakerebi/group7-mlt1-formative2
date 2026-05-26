from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ["tables_dir", "figures_dir", "predictions_dir", "model_dir", "embedding_dir"]:
        ensure_dir(config["paths"][key])
