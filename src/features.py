from __future__ import annotations

from sklearn.pipeline import FeatureUnion
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


def char_tfidf(
    ngram_range: tuple[int, int] = (3, 5),
    max_features: int | None = 20_000,
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def combined_word_char_tfidf(
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_max_features: int | None = 25_000,
    char_max_features: int | None = 25_000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", word_tfidf(word_ngram_range, word_max_features, min_df=2, max_df=0.95)),
            ("char", char_tfidf(char_ngram_range, char_max_features, min_df=2, max_df=0.95)),
        ]
    )
