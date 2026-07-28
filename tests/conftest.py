"""Shared fixtures.

Every test runs on a tiny synthetic corpus, not the real 20 Newsgroups download.
That is deliberate: the suite must pass in CI without fetching 15 MB, and a test
that depends on a dataset server being up is a test that eventually fails for
reasons unrelated to the code.

The synthetic corpus has genuinely separable topics (distinct vocabularies), so
a fitted pipeline reaches near-perfect accuracy on it - which lets the tests
assert real behaviour ("this learns and explains"), not merely "it does not
raise".
"""

from __future__ import annotations

import random

import pytest
from sklearn.pipeline import Pipeline

from news_classifier import model

TOPIC_VOCAB = {
    "space": "space orbit rocket nasa launch planet astronaut satellite mission moon",
    "sports": "team game score player coach league season goal championship match",
    "cooking": "recipe oven flour sugar bake dough oven butter simmer sauce",
    "finance": "market stock invest bond dividend portfolio interest rate broker equity",
}
TOPIC_NAMES = sorted(TOPIC_VOCAB)


def _make_document(rng: random.Random, words: str, n: int = 20) -> str:
    pool = words.split()
    return " ".join(rng.choice(pool) for _ in range(n))


@pytest.fixture(scope="session")
def synthetic_corpus():
    """A separable 4-topic corpus: (X_train, y_train, X_test, y_test, names)."""
    rng = random.Random(42)

    X_train, y_train, X_test, y_test = [], [], [], []
    for label, name in enumerate(TOPIC_NAMES):
        vocab = TOPIC_VOCAB[name]
        for _ in range(40):
            X_train.append(_make_document(rng, vocab))
            y_train.append(label)
        for _ in range(15):
            X_test.append(_make_document(rng, vocab))
            y_test.append(label)

    return X_train, y_train, X_test, y_test, TOPIC_NAMES


@pytest.fixture
def fitted_pipeline(synthetic_corpus) -> Pipeline:
    """A pipeline trained on the synthetic corpus, ready to predict and explain."""
    X_train, y_train, _, _, _ = synthetic_corpus

    # min_df=1: the synthetic vocabulary is tiny, so the real default would drop
    # almost everything. The production default is validated separately.
    pipeline = model.build_pipeline()
    pipeline.named_steps["tfidf"].set_params(min_df=1, stop_words=None, ngram_range=(1, 1))
    pipeline.fit(X_train, y_train)
    return pipeline
