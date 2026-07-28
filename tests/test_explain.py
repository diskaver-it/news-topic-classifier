"""Tests for the interpretability layer - the project's distinguishing feature."""

from __future__ import annotations

import pytest

from news_classifier import explain
from tests.conftest import TOPIC_VOCAB


class TestTopFeaturesPerClass:
    def test_returns_entries_for_every_class(self, fitted_pipeline, synthetic_corpus):
        _, _, _, _, names = synthetic_corpus

        top = explain.top_features_per_class(fitted_pipeline, names, top_k=5)

        assert set(top) == set(names)
        assert all(len(top[name]) == 5 for name in names)

    def test_defining_words_come_from_the_topic_vocabulary(
        self, fitted_pipeline, synthetic_corpus
    ):
        """The top word for 'space' must actually be a space word, etc.

        This is the real assertion: the model's learned weights must reflect the
        subject matter, not arbitrary tokens.
        """
        _, _, _, _, names = synthetic_corpus
        top = explain.top_features_per_class(fitted_pipeline, names, top_k=3)

        for topic in names:
            topic_words = set(TOPIC_VOCAB[topic].split())
            top_tokens = {item["token"] for item in top[topic]}
            assert top_tokens & topic_words, f"{topic}: {top_tokens} vs {topic_words}"

    def test_weights_are_sorted_descending(self, fitted_pipeline, synthetic_corpus):
        _, _, _, _, names = synthetic_corpus
        top = explain.top_features_per_class(fitted_pipeline, names, top_k=5)

        for items in top.values():
            weights = [item["weight"] for item in items]
            assert weights == sorted(weights, reverse=True)


class TestExplainPrediction:
    def test_predicts_the_right_topic_with_a_bundle(self, fitted_pipeline, synthetic_corpus):
        _, _, _, _, names = synthetic_corpus
        text = TOPIC_VOCAB["space"]

        result = explain.explain_prediction(fitted_pipeline, text, names)

        assert result["predicted_topic"] == "space"
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["runner_up_topic"] != result["predicted_topic"]
        assert set(result["probabilities"]) == set(names)

    def test_probabilities_sum_to_one(self, fitted_pipeline, synthetic_corpus):
        _, _, _, _, names = synthetic_corpus

        result = explain.explain_prediction(fitted_pipeline, TOPIC_VOCAB["finance"], names)

        assert sum(result["probabilities"].values()) == pytest.approx(1.0, rel=1e-6)

    def test_top_tokens_actually_occur_in_the_input(self, fitted_pipeline, synthetic_corpus):
        """A contribution can only come from a word present in the document."""
        _, _, _, _, names = synthetic_corpus
        text = "rocket orbit nasa launch"

        result = explain.explain_prediction(fitted_pipeline, text, names)

        input_words = set(text.split())
        for item in result["top_contributing_tokens"]:
            assert item["token"] in input_words

    def test_top_contribution_pushes_toward_the_prediction(
        self, fitted_pipeline, synthetic_corpus
    ):
        """For a clear document the strongest contributor must be positive."""
        _, _, _, _, names = synthetic_corpus

        result = explain.explain_prediction(fitted_pipeline, TOPIC_VOCAB["sports"], names)

        assert result["top_contributing_tokens"][0]["contribution"] > 0

    def test_handles_out_of_vocabulary_text_without_raising(
        self, fitted_pipeline, synthetic_corpus
    ):
        """Words the model never saw simply contribute nothing - no crash."""
        _, _, _, _, names = synthetic_corpus

        result = explain.explain_prediction(fitted_pipeline, "zzz qqq unknowntoken", names)

        assert result["predicted_topic"] in names
        assert result["top_contributing_tokens"] == []   # nothing recognised
