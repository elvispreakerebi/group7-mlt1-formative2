from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def word_tfidf(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 20_000,
    min_df: int = 1,
    max_df: float = 1.0,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
