"""Tests for the pipeline construction and its learning behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from news_classifier import config, model


class TestBuildPipeline:
    def test_has_the_two_expected_steps(self):
        pipeline = model.build_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["tfidf", "clf"]

    def test_passes_through_the_configured_C(self):
        pipeline = model.build_pipeline(C=7.0)
        assert pipeline.named_steps["clf"].C == 7.0

    def test_vectoriser_uses_the_configured_ngram_range(self):
        pipeline = model.build_pipeline()
        assert pipeline.named_steps["tfidf"].ngram_range == config.TFIDF_NGRAM_RANGE


class TestLearning:
    def test_separates_distinct_topics(self, fitted_pipeline, synthetic_corpus):
        """On a separable corpus the pipeline should be near-perfect."""
        _, _, X_test, y_test, _ = synthetic_corpus

        accuracy = (np.array(fitted_pipeline.predict(X_test)) == np.array(y_test)).mean()
        assert accuracy > 0.9

    def test_predict_proba_rows_are_distributions(self, fitted_pipeline, synthetic_corpus):
        _, _, X_test, _, names = synthetic_corpus

        proba = fitted_pipeline.predict_proba(X_test)

        assert proba.shape == (len(X_test), len(names))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_coefficient_matrix_is_one_row_per_class(self, fitted_pipeline, synthetic_corpus):
        _, _, _, _, names = synthetic_corpus

        coef = fitted_pipeline.named_steps["clf"].coef_
        n_features = len(fitted_pipeline.named_steps["tfidf"].get_feature_names_out())

        assert coef.shape == (len(names), n_features)


@pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
def test_vectoriser_handles_effectively_empty_text(fitted_pipeline, empty):
    """A blank document must vectorise to all-zeros, not raise."""
    row = fitted_pipeline.named_steps["tfidf"].transform([empty])
    assert row.nnz == 0
