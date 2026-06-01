from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import prepare_text
from src.utils import ensure_dir, resolve_path


def sentence_embedding_cache_path(model_name: str, split_name: str, embedding_dir: str | Path) -> Path:
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return resolve_path(embedding_dir) / f"{safe_name}_{split_name}.npy"


def encode_with_sentence_transformer(
    frame: pd.DataFrame,
    model_name: str,
    embedding_dir: str | Path,
    split_name: str,
    use_type_token: bool = True,
    text_col: str = "Text",
    type_col: str = "Type",
) -> np.ndarray:
    cache_path = sentence_embedding_cache_path(model_name, split_name, embedding_dir)
    if cache_path.exists():
        return np.load(cache_path)

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from sentence_transformers import SentenceTransformer

    ensure_dir(cache_path.parent)
    text = prepare_text(frame, use_type_token=use_type_token, text_col=text_col, type_col=type_col).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_path, embeddings)
    return embeddings
