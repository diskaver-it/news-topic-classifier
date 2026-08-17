"""Confidence: is a "99%" worth believing, and when should the model refuse?

The demo shows a probability next to every prediction, and until now that number
had never been checked. `predict_proba` returns whatever the softmax over the
decision function happens to say - a quantity that orders documents correctly
while being systematically over- or under-confident about all of them. On a
20-class problem trained with L2 regularisation it is usually *under*-confident,
which is the less famous direction and just as wrong.

Two things follow from measuring it, and they are the same feature seen twice:

  * **calibration.** Among the posts the model calls 80% confident, roughly 80%
    should be right. That is checkable, and temperature scaling fixes it with a
    single parameter - no retraining, no change to the ranking, so accuracy and
    every explanation stay exactly as they were.

  * **abstention.** Once the confidence means something, it can be used as a
    decision: answer when confident, hand the rest to a human. A topic router
    that is 70% accurate on everything is much less useful than one that is 90%
    accurate on the three quarters of posts it is sure about - and says so about
    the rest. Without calibration that threshold is unpickable.

The order matters. Selecting on an uncalibrated confidence still works, because
selection only needs the ranking; but the *threshold* someone picks - "answer
above 0.9" - means nothing until the number does.
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

    The max is subtracted before exponentiating. That is not a micro
    optimisation: `exp(800)` overflows to infinity and the row comes back as
    NaN, and subtracting a per-row constant leaves the result unchanged because
    the constant cancels between numerator and denominator.
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

    Temperature scaling divides every logit by one number, chosen to minimise
    the negative log-likelihood. Because it is a monotone transformation applied
    identically to all classes, the arg-max never moves: **accuracy, macro-F1,
    the confusion matrix and every word-level explanation are unchanged**. Only
    the confidence attached to the answer changes. That is what makes it safe to
    bolt on to a model that has already been evaluated.

    T > 1 softens over-confident probabilities; T < 1 sharpens under-confident
    ones. Regularised linear models on sparse text usually land below 1.

    The objective is smooth and one-dimensional, so a bounded scalar search is
    both sufficient and the honest amount of machinery for the job.
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

    ECE is the number to quote and MCE is the number to worry about. A model can
    have a respectable ECE while being wildly wrong in one sparsely populated
    bin, and if that bin is the high-confidence one - the only one anybody acts
    on - the average has hidden the thing that matters.

    Reported alongside: the mean confidence and the actual accuracy. If mean
    confidence is below accuracy the model is *under*-confident, which is the
    usual direction for a regularised linear model and the opposite of the
    over-confidence neural networks are known for.
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

    Coverage is the share of documents the model is willing to answer at all;
    selective accuracy is how often it is right *among those*. The trade-off is
    the whole product decision: a router that answers everything at 0.70 and one
    that answers 60% of the traffic at 0.90 are different systems, and which is
    better depends on what the other 40% costs to review.

    A threshold with no documents above it is reported as zero coverage and an
    undefined accuracy of 0.0 rather than a NaN, so the curve stays plottable.
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

    "Most permissive" is the point: any threshold high enough gets there by
    answering three documents, so the useful question is the *lowest* cutoff
    that clears the bar, because that is the one that answers the most traffic.
    `min_coverage` refuses the degenerate answers.

    Must be chosen on data the model has not been evaluated on if the resulting
    coverage is to be believed - the same rule as any other tuned threshold.
    Returns `achievable: False` rather than a made-up cutoff when the target is
    out of reach, because silently returning 0.99 would look like a result.
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

    Goes through `decision_function` rather than `predict_proba` because the
    temperature belongs on the logits; re-deriving them from probabilities would
    be the same arithmetic done backwards and less precisely.

    Multiclass only. A binary classifier's `decision_function` returns a single
    column, and a softmax over one column is identically 1.0 - so this refuses
    rather than returning a confidently meaningless answer.
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
