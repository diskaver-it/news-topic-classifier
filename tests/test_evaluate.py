"""Tests for the evaluation and confusion-analysis helpers."""

from __future__ import annotations

import numpy as np

from news_classifier import config, evaluate


class TestEvaluatePredictions:
    def test_perfect_predictions_score_one(self):
        y = [0, 1, 2, 0, 1, 2]
        names = ["a", "b", "c"]

        metrics = evaluate.evaluate_predictions(y, y, names)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

    def test_reports_every_class_in_the_breakdown(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        names = ["a", "b", "c"]

        metrics = evaluate.evaluate_predictions(y_true, y_pred, names)

        for name in names:
            assert name in metrics["per_class"]
            assert "f1-score" in metrics["per_class"][name]

    def test_macro_f1_penalises_a_failed_minority_class(self):
        """Macro-F1 must drop when a small class is missed, even if accuracy stays high."""
        # 10 of class 0 (all correct), 2 of class 1 (both wrong).
        y_true = [0] * 10 + [1, 1]
        y_pred = [0] * 10 + [0, 0]
        names = ["major", "minor"]

        metrics = evaluate.evaluate_predictions(y_true, y_pred, names)

        assert metrics["accuracy"] > 0.8           # the majority carries accuracy
        assert metrics["macro_f1"] < 0.5           # but macro-F1 exposes the failure


class TestTopConfusions:
    def test_finds_the_most_frequent_mistake(self):
        names = ["a", "b", "c"]
        # Five a->b confusions, nothing else wrong.
        y_true = [0, 0, 0, 0, 0, 1, 2]
        y_pred = [1, 1, 1, 1, 1, 1, 2]

        confusions = evaluate.top_confusions(y_true, y_pred, names, k=5)

        assert confusions[0]["true"] == "a"
        assert confusions[0]["predicted"] == "b"
        assert confusions[0]["count"] == 5

    def test_ignores_correct_predictions(self):
        names = ["a", "b"]
        confusions = evaluate.top_confusions([0, 1, 0, 1], [0, 1, 0, 1], names)
        assert confusions == []          # a perfect classifier has no confusions

    def test_flags_sibling_topic_confusions(self):
        names = list(config.SUPERCATEGORIES)      # the real 20 topic names
        i_pc = names.index("comp.sys.ibm.pc.hardware")
        i_mac = names.index("comp.sys.mac.hardware")

        confusions = evaluate.top_confusions([i_pc] * 3, [i_mac] * 3, names, k=1)

        assert confusions[0]["same_supercategory"] is True   # both are computers


class TestNormalisedConfusion:
    def test_rows_sum_to_one_for_present_classes(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        matrix = evaluate.confusion_matrix_normalised(y_true, y_pred)

        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
