"""The classification pipeline: TF-IDF vectoriser + logistic-regression head.

Why a linear model on TF-IDF, and not something heavier:

  * It is the honest strong baseline for topic classification. On 20 Newsgroups
    a well-tuned linear model lands within a few points of far more expensive
    approaches, so anything fancier has to justify its cost - and for a
    portfolio piece, "I reached for a transformer first" is the wrong instinct
    to show.
  * It is interpretable *per prediction*. Because the decision for class c is a
    linear function of the token weights, we can point at the exact words that
    pushed a document toward its predicted topic (see explain.py). A gradient
    boosting model or a neural net cannot hand you that as cleanly.
  * It trains in seconds and the whole artifact is a few MB, so the Streamlit
    demo loads instantly.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


def build_pipeline(C: float = config.LOGREG_C) -> Pipeline:
    """TF-IDF followed by multinomial logistic regression.

    Vectoriser choices worth defending:
      * `sublinear_tf=True` replaces raw counts with 1 + log(count). A word
        appearing 10 times is more informative than one appearing once, but not
        ten times as informative - the log tames that.
      * `ngram_range=(1, 2)` adds bigrams, so "new york" or "hard drive" become
        features in their own right rather than dissolving into single words.
      * `min_df` / `max_df` trim the long tail of typos and the corpus-wide
        boilerplate that slips past the stop-word list.
      * `strip_accents` and lowercasing normalise trivial surface variation.

    Classifier choices:
      * `C` is the inverse regularisation strength, tuned by cross-validation in
        train.py rather than pinned by hand.
      * With 20 classes and tens of thousands of sparse features, L2-regularised
        logistic regression is both fast and well-behaved. `max_iter` is raised
        so the solver actually converges on this many features.

    The two steps are one Pipeline so the fitted vocabulary and the model travel
    together in a single joblib file - there is no separate "remember to run the
    vectoriser first" step at serving time.
    """
    vectoriser = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        sublinear_tf=True,
        ngram_range=config.TFIDF_NGRAM_RANGE,
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
