"""Confidence: is a "99%" worth believing, and when should the model refuse?

The demo shows a probability next to every prediction and that number had never
been checked. A softmax over the decision function orders documents correctly
while being systematically over- or under-confident about all of them - on 20
classes with L2 regularisation, usually under.

Two things follow, and they are the same feature twice. Calibration: among the
posts called 80% confident, about 80% should be right, and temperature scaling
fixes that with one parameter, no retraining and no change to the ranking.
Abstention: once the confidence means something it can be a decision - answer
when sure, hand the rest to a human.

The order matters. Selection only needs the ranking, so it works on an
uncalibrated score; but a threshold someone picks - "answer above 0.9" - means
nothing until the number does.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Softmax with a temperature
# --------------------------------------------------------------------------
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of `logits / temperature`.

    The per-row max comes off before exponentiating - not an optimisation:
    exp(800) overflows and the row returns NaN. The constant cancels between
    numerator and denominator, so the result is unchanged.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    scaled = np.asarray(logits, dtype=float) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray, y: Sequence[int], temperature: float
) -> float:
    """Mean NLL of the true labels under the temperature-scaled softmax."""
    probabilities = softmax(logits, temperature)
    true_class = probabilities[np.arange(len(y)), np.asarray(y, dtype=int)]
    return float(-np.mean(np.log(np.clip(true_class, 1e-12, None))))


def fit_temperature(
    logits: np.ndarray,
    y: Sequence[int],
    bounds: tuple = (0.05, 10.0),
) -> float:
    """Find the single scalar that best calibrates the model (Guo et al., 2017).

    One number divides every logit, chosen to minimise the NLL. Monotone and
    applied identically to all classes, so the arg-max never moves: accuracy,
    macro-F1, the confusion matrix and every explanation are unchanged. That is
    what makes it safe to bolt onto an already-evaluated model.

    T > 1 softens over-confidence, T < 1 sharpens under-confidence; regularised
    linear models on sparse text usually land below 1.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError(f"expected a 2-D logit matrix, got shape {logits.shape}")
    if len(logits) != len(y):
        raise ValueError("logits and labels must line up")

    result = minimize_scalar(
        lambda t: negative_log_likelihood(logits, y, t),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


# --------------------------------------------------------------------------
# How wrong were the probabilities?
# --------------------------------------------------------------------------
def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = config.CALIBRATION_BINS,
) -> List[Dict[str, float]]:
    """Group predictions by confidence and compare each group's claim to reality.

    Equal-width bins on [0, 1] rather than equal-count ones, because the picture
    this feeds is a reliability diagram, and the diagonal it is compared against
    is only meaningful on a fixed confidence axis.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []

    for low, high in zip(edges[:-1], edges[1:]):
        # Upper-inclusive on the last bin so a confidence of exactly 1.0 counts.
        in_bin = (confidences > low) & (confidences <= high)
        if high == edges[-1]:
            in_bin |= confidences == low

        count = int(in_bin.sum())
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "count": count,
                "mean_confidence": float(confidences[in_bin].mean()) if count else 0.0,
                "accuracy": float(correct[in_bin].mean()) if count else 0.0,
            }
        )

    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = config.CALIBRATION_BINS,
) -> Dict[str, float]:
    """ECE and MCE: the average and the worst gap between claim and reality.

        ECE = sum over bins of  (n_bin / n) * |accuracy(bin) - confidence(bin)|

    ECE is the number to quote, MCE the one to worry about: a respectable ECE
    can hide one sparse bin that is wildly wrong, and if that is the
    high-confidence bin it is the only one anybody acts on.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    bins = reliability_bins(confidences, correct, n_bins)

    total = max(len(confidences), 1)
    gaps = [
        (b["count"] / total, abs(b["accuracy"] - b["mean_confidence"]))
        for b in bins
        if b["count"] > 0
    ]

    return {
        "ece": float(sum(weight * gap for weight, gap in gaps)),
        "mce": float(max((gap for _, gap in gaps), default=0.0)),
        "mean_confidence": float(confidences.mean()),
        "accuracy": float(correct.mean()),
        "overconfident": bool(confidences.mean() > correct.mean()),
        "n_bins": int(n_bins),
    }


# --------------------------------------------------------------------------
# Answering only when confident
# --------------------------------------------------------------------------
def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Accuracy against coverage as the abstention threshold moves.

    The whole product decision: a router answering everything at 0.70 and one
    answering 60% of traffic at 0.90 are different systems, and which is better
    depends on what the other 40% costs to review.

    An empty threshold reports zero coverage and accuracy 0.0 rather than NaN,
    so the curve stays plottable.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    if thresholds is None:
        thresholds = np.linspace(0.0, 0.99, 100)

    curve = []
    for threshold in thresholds:
        answered = confidences >= threshold
        n_answered = int(answered.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(n_answered / max(len(confidences), 1)),
                "n_answered": n_answered,
                "selective_accuracy": (
                    float(correct[answered].mean()) if n_answered else 0.0
                ),
            }
        )

    return curve


def threshold_for_target_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target_accuracy: float = config.ABSTAIN_TARGET_ACCURACY,
    min_coverage: float = 0.05,
) -> Dict[str, float]:
    """The most permissive threshold that still reaches the target accuracy.

    "Most permissive" is the point - any high enough cutoff gets there by
    answering three documents, so the useful one is the lowest that clears the
    bar. `min_coverage` refuses the degenerate answers, and an unreachable
    target returns `achievable: False` rather than a 0.99 that looks like one.
    """
    curve = risk_coverage_curve(confidences, correct)
    feasible = [
        point for point in curve
        if point["selective_accuracy"] >= target_accuracy
        and point["coverage"] >= min_coverage
    ]

    if not feasible:
        return {
            "target_accuracy": float(target_accuracy),
            "achievable": False,
            "note": (
                "No threshold reaches this accuracy while still answering "
                f"{100 * min_coverage:.0f}% of documents."
            ),
        }

    best = min(feasible, key=lambda point: point["threshold"])
    return {
        "target_accuracy": float(target_accuracy),
        "achievable": True,
        "threshold": best["threshold"],
        "coverage": best["coverage"],
        "selective_accuracy": best["selective_accuracy"],
        "n_answered": best["n_answered"],
    }


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------
def calibrated_probabilities(
    pipeline, texts: Sequence[str], temperature: float = 1.0
) -> np.ndarray:
    """`predict_proba` with the fitted temperature applied.

    Through `decision_function`, because the temperature belongs on the logits;
    recovering them from probabilities is the same arithmetic backwards and
    less precisely. Multiclass only - a softmax over one column is always 1.0.
    """
    logits = np.asarray(pipeline.decision_function(list(texts)), dtype=float)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            "temperature scaling here assumes a multiclass decision_function; "
            f"got shape {logits.shape}"
        )
    return softmax(logits, temperature)


def evaluate_calibration(
    logits: np.ndarray,
    y: Sequence[int],
    temperature: float,
    n_bins: int = config.CALIBRATION_BINS,
) -> Dict:
    """Before-and-after report for one set of logits.

    Both arms are computed from the same logits, so the comparison isolates the
    temperature and nothing else. The accuracy is asserted to be identical in
    both - if it ever differs, the scaling has been applied wrongly.
    """
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)

    report: Dict[str, Dict] = {}
    for label, t in (("uncalibrated", 1.0), ("calibrated", temperature)):
        probabilities = softmax(logits, t)
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = predicted == y

        report[label] = {
            "temperature": float(t),
            **expected_calibration_error(confidence, correct, n_bins),
            "nll": negative_log_likelihood(logits, y, t),
        }

    report["ece_reduction"] = round(
        report["uncalibrated"]["ece"] - report["calibrated"]["ece"], 4
    )
    report["accuracy_unchanged"] = bool(
        report["uncalibrated"]["accuracy"] == report["calibrated"]["accuracy"]
    )
    return report
