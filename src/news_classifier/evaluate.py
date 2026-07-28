"""Evaluation metrics and confusion analysis for the multiclass classifier."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
) -> Dict:
    """Headline metrics plus a full per-class breakdown.

    Macro-F1 is reported alongside accuracy on purpose. Accuracy weights every
    document equally, so a model can look good by nailing the big, easy classes
    and quietly failing a small one. Macro-F1 averages the per-class F1 with
    equal weight *per class*, so a topic the model cannot handle drags the score
    down no matter how rare it is - which is what you want to know.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    per_class = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def top_confusions(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
    k: int = 10,
) -> List[Dict]:
    """The k most frequent (true -> predicted) mistakes.

    A raw 20x20 confusion matrix is hard to read; the pairs the model actually
    confuses tell the story faster. The `same_supercategory` flag is the point:
    most errors are between sibling topics (two comp.* groups, or atheism vs
    christianity), which is the model being *reasonably* wrong rather than
    cluelessly wrong.
    """
    # Pin the matrix to the full label set. Without `labels`, confusion_matrix
    # sizes itself to whichever classes happen to appear in this batch, and the
    # row/column indices would stop lining up with target_names.
    labels = list(range(len(target_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(matrix, 0)     # keep only the mistakes

    confusions = []
    for true_idx, pred_idx in zip(*np.unravel_index(
        np.argsort(matrix, axis=None)[::-1], matrix.shape
    )):
        count = int(matrix[true_idx, pred_idx])
        if count == 0:
            break
        true_name = target_names[true_idx]
        pred_name = target_names[pred_idx]
        confusions.append({
            "true": true_name,
            "predicted": pred_name,
            "count": count,
            "same_supercategory": (
                config.SUPERCATEGORIES.get(true_name)
                == config.SUPERCATEGORIES.get(pred_name)
            ),
        })
        if len(confusions) == k:
            break

    return confusions


def confusion_matrix_normalised(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Row-normalised confusion matrix (each row = true class, sums to 1).

    Row normalisation answers "of the posts that really are topic X, what share
    did the model send where?" - readable regardless of how many test documents
    each topic happens to have.
    """
    matrix = confusion_matrix(y_true, y_pred).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0        # guard against an empty class
    return matrix / row_sums
