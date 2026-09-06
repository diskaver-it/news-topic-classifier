"""The classification pipeline: TF-IDF vectoriser + logistic-regression head.

Linear on TF-IDF rather than something heavier for one reason that matters and
two that help: the decision for class c is a linear function of the token
weights, so the exact words that pushed a document toward its topic can be
named (see explain.py). On top of that it lands within a few points of far more
expensive approaches on this corpus, and the artifact is small enough that the
Streamlit demo loads instantly.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


def build_pipeline(C: float = config.LOGREG_C) -> Pipeline:
    """TF-IDF followed by multinomial logistic regression.

    One Pipeline so the fitted vocabulary and the model travel together in a
    single joblib file, with no "remember to run the vectoriser first" at
    serving time. `C` is tuned by cross-validation in train.py.
    """
    vectoriser = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        # A word appearing 10 times is more informative than one appearing
        # once, but not ten times as informative.
        sublinear_tf=True,
        # Bigrams so "hard drive" is a feature rather than two dissolved words.
        ngram_range=config.TFIDF_NGRAM_RANGE,
        # Trim the typo tail and the boilerplate that gets past the stop words.
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        max_features=config.TFIDF_MAX_FEATURES,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )

    return Pipeline(steps=[("tfidf", vectoriser), ("clf", classifier)])
