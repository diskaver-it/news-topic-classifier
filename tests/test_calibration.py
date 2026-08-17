"""Tests for calibration and abstention.

Calibration code is easy to write and easy to get subtly wrong: a temperature
applied to probabilities instead of logits, a reliability bin that silently
drops the top of the range, an abstention threshold picked on the data it is
then evaluated on. Each test below pins one of those.

The two invariants worth stating up front, because they are what makes
temperature scaling safe to add to an already-evaluated model:

  * scaling never changes which class wins, so accuracy is untouched;
  * on data generated with a known miscalibration, the fitted temperature must
    recover it and drive the calibration error down.
"""

from __future__ import annotations

import numpy as np
import pytest

from news_classifier import calibration


@pytest.fixture
def miscalibrated():
    """Logits with a known temperature baked in, plus labels drawn from them.

    Labels are sampled from the *true* probabilities, so the correct answer is
    known by construction: a temperature of `true_t` should be recovered, and
    scaling by it should make the confidences honest.
    """
    rng = np.random.default_rng(0)
    n, n_classes, true_t = 4_000, 6, 2.5

    true_logits = rng.normal(0, 2.0, size=(n, n_classes))
    probabilities = calibration.softmax(true_logits, 1.0)
    y = np.array([rng.choice(n_classes, p=row) for row in probabilities])

    # What the "model" reports: the same ordering, sharpened by true_t.
    reported_logits = true_logits * true_t
    return reported_logits, y, true_t


class TestSoftmax:
    def test_rows_sum_to_one(self):
        probabilities = calibration.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))

        assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])

    def test_survives_logits_that_would_overflow(self):
        """exp(800) is inf; subtracting the row max is what stops the NaN."""
        probabilities = calibration.softmax(np.array([[800.0, 799.0, -800.0]]))

        assert np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0)

    def test_temperature_moves_confidence_without_moving_the_winner(self):
        logits = np.array([[3.0, 1.0, 0.5]])

        sharp = calibration.softmax(logits, 0.5)
        soft = calibration.softmax(logits, 5.0)

        assert sharp.argmax() == soft.argmax()
        assert sharp.max() > soft.max()

    def test_rejects_a_non_positive_temperature(self):
        with pytest.raises(ValueError):
            calibration.softmax(np.array([[1.0, 2.0]]), temperature=0.0)


class TestFitTemperature:
    def test_recovers_a_known_miscalibration(self, miscalibrated):
        logits, y, true_t = miscalibrated

        fitted = calibration.fit_temperature(logits, y)

        assert fitted == pytest.approx(true_t, rel=0.15)

    def test_leaves_already_calibrated_logits_alone(self, miscalibrated):
        logits, y, true_t = miscalibrated

        fitted = calibration.fit_temperature(logits / true_t, y)

        assert fitted == pytest.approx(1.0, rel=0.15)

    def test_the_fitted_temperature_minimises_the_objective(self, miscalibrated):
        logits, y, _ = miscalibrated
        fitted = calibration.fit_temperature(logits, y)

        best = calibration.negative_log_likelihood(logits, y, fitted)
        for other in (fitted * 0.7, fitted * 1.3):
            assert calibration.negative_log_likelihood(logits, y, other) > best

    def test_rejects_misshapen_input(self):
        with pytest.raises(ValueError):
            calibration.fit_temperature(np.array([1.0, 2.0, 3.0]), [0, 1, 2])
        with pytest.raises(ValueError):
            calibration.fit_temperature(np.zeros((3, 4)), [0, 1])


class TestCalibrationError:
    def test_a_perfect_predictor_has_no_calibration_error(self):
        confidences = np.full(200, 1.0)
        correct = np.full(200, True)

        result = calibration.expected_calibration_error(confidences, correct)

        assert result["ece"] == pytest.approx(0.0)

    def test_confident_and_wrong_is_the_worst_case(self):
        confidences = np.full(200, 1.0)
        correct = np.full(200, False)

        result = calibration.expected_calibration_error(confidences, correct)

        assert result["ece"] == pytest.approx(1.0)
        assert result["overconfident"] is True

    def test_detects_under_confidence_too(self):
        """The direction a regularised linear model usually errs in."""
        confidences = np.full(200, 0.30)
        correct = np.full(200, True)

        result = calibration.expected_calibration_error(confidences, correct)

        assert result["ece"] == pytest.approx(0.70)
        assert result["overconfident"] is False

    def test_bins_account_for_every_prediction(self):
        rng = np.random.default_rng(3)
        confidences = rng.uniform(0, 1, 500)
        correct = rng.random(500) < confidences

        bins = calibration.reliability_bins(confidences, correct)

        assert sum(b["count"] for b in bins) == 500

    def test_mce_is_at_least_ece(self):
        rng = np.random.default_rng(4)
        confidences = rng.uniform(0.5, 1.0, 500)
        correct = rng.random(500) < 0.6

        result = calibration.expected_calibration_error(confidences, correct)

        assert result["mce"] >= result["ece"]


class TestEndToEnd:
    def test_scaling_lowers_the_calibration_error_and_not_the_accuracy(
        self, miscalibrated
    ):
        logits, y, _ = miscalibrated
        temperature = calibration.fit_temperature(logits, y)

        report = calibration.evaluate_calibration(logits, y, temperature)

        assert report["calibrated"]["ece"] < report["uncalibrated"]["ece"]
        assert report["ece_reduction"] > 0
        # The point of temperature scaling: the arg-max never moves.
        assert report["accuracy_unchanged"] is True
        assert report["calibrated"]["nll"] <= report["uncalibrated"]["nll"]


class TestAbstention:
    @staticmethod
    def _graded(n=2_000, seed=5):
        """Confidence that genuinely predicts correctness, as a real model's does."""
        rng = np.random.default_rng(seed)
        confidences = rng.uniform(0.2, 1.0, n)
        correct = rng.random(n) < confidences
        return confidences, correct

    def test_accuracy_rises_as_coverage_falls(self):
        confidences, correct = self._graded()

        curve = calibration.risk_coverage_curve(confidences, correct)
        answered = [p for p in curve if p["coverage"] > 0.05]

        assert answered[0]["coverage"] > answered[-1]["coverage"]
        assert answered[-1]["selective_accuracy"] > answered[0]["selective_accuracy"]

    def test_zero_threshold_answers_everything(self):
        confidences, correct = self._graded()

        curve = calibration.risk_coverage_curve(confidences, correct, thresholds=[0.0])

        assert curve[0]["coverage"] == pytest.approx(1.0)
        assert curve[0]["selective_accuracy"] == pytest.approx(correct.mean())

    def test_picks_the_most_permissive_threshold_that_clears_the_bar(self):
        confidences, correct = self._graded()

        point = calibration.threshold_for_target_accuracy(
            confidences, correct, target_accuracy=0.85
        )

        assert point["achievable"] is True
        assert point["selective_accuracy"] >= 0.85
        # Anything lower would have failed the target - that is what "most
        # permissive" means, and it is what maximises coverage.
        stricter = calibration.risk_coverage_curve(
            confidences, correct, thresholds=[point["threshold"] - 0.02]
        )[0]
        assert stricter["selective_accuracy"] < 0.85

    def test_says_so_instead_of_inventing_a_threshold(self):
        """A target no cutoff can reach must fail loudly, not return 0.99."""
        rng = np.random.default_rng(6)
        confidences = rng.uniform(0.2, 1.0, 1_000)
        correct = rng.random(1_000) < 0.5      # confidence carries no information

        point = calibration.threshold_for_target_accuracy(
            confidences, correct, target_accuracy=0.99
        )

        assert point["achievable"] is False
        assert "threshold" not in point

    def test_a_coverage_floor_can_rule_the_target_out(self):
        """Reaching 85% here means answering ~37% of documents.

        Asked for the same accuracy while covering at least half the traffic,
        the honest answer is that it cannot be done - not a threshold that
        quietly ignores the floor.
        """
        confidences, correct = self._graded()

        permissive = calibration.threshold_for_target_accuracy(
            confidences, correct, target_accuracy=0.85, min_coverage=0.05
        )
        with_floor = calibration.threshold_for_target_accuracy(
            confidences, correct, target_accuracy=0.85, min_coverage=0.5
        )

        assert permissive["achievable"] is True
        assert permissive["coverage"] < 0.5
        assert with_floor["achievable"] is False


class TestServing:
    def test_calibrated_probabilities_match_the_pipeline(self, fitted_pipeline):
        """The demo path and the training path must compute the same number."""
        texts = ["space orbit rocket nasa launch", "recipe oven flour sugar bake"]

        probabilities = calibration.calibrated_probabilities(
            fitted_pipeline, texts, temperature=1.0
        )

        assert probabilities.shape[0] == 2
        assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
        assert list(probabilities.argmax(axis=1)) == list(fitted_pipeline.predict(texts))

    def test_temperature_does_not_change_the_predicted_topic(self, fitted_pipeline):
        texts = ["team game score player coach league", "market stock invest bond"]

        hot = calibration.calibrated_probabilities(fitted_pipeline, texts, 3.0)
        cold = calibration.calibrated_probabilities(fitted_pipeline, texts, 0.4)

        assert list(hot.argmax(axis=1)) == list(cold.argmax(axis=1))
        assert (cold.max(axis=1) >= hot.max(axis=1)).all()
