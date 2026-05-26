from __future__ import annotations

import re

import pandas as pd


SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def clean_text(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = text.replace("\x00", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip().lower()


def add_type_token(frame: pd.DataFrame, text_col: str = "Text", type_col: str = "Type") -> pd.Series:
    type_tokens = frame[type_col].fillna("unknown").astype(str).str.lower().str.replace(r"\W+", "_", regex=True)
    cleaned = frame[text_col].map(clean_text)
    return "type_" + type_tokens + " " + cleaned


def prepare_text(frame: pd.DataFrame, use_type_token: bool = False, text_col: str = "Text", type_col: str = "Type") -> pd.Series:
    if use_type_token:
        return add_type_token(frame, text_col=text_col, type_col=type_col)
    return frame[text_col].map(clean_text)
