"""Interpretability: which words drove a prediction, and what defines a topic.

The part that makes a linear model worth choosing. The score for class c is

    score_c = bias_c + sum over tokens t of  tfidf(t) * weight[c, t]

and every term of that sum is one word's signed contribution to one class -
which gives both the global view (the words defining a topic) and the local one
(what pushed *this* document to its predicted topic).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.pipeline import Pipeline

from . import calibration, config


def top_features_per_class(
    pipeline: Pipeline,
    target_names: List[str],
    top_k: int = config.TOP_FEATURES_PER_TOPIC,
) -> Dict[str, List[Dict[str, float]]]:
    """The tokens with the largest positive weight for each class.

    The model's learned definition of each topic, and a sanity check: launch,
    orbit, nasa for sci.space means it learned the subject; function words
    would mean something is wrong with the vectoriser.
    """
    vectoriser = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    feature_names = np.asarray(vectoriser.get_feature_names_out())
    # coef_ has shape (n_classes, n_features); binary problems collapse to one
    # row, but 20 Newsgroups is always multiclass so this stays (20, n_features).
    coefficients = classifier.coef_

    result: Dict[str, List[Dict[str, float]]] = {}
    for class_idx, class_name in enumerate(target_names):
        weights = coefficients[class_idx]
        top_idx = np.argsort(weights)[::-1][:top_k]
        result[class_name] = [
            {"token": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_idx
        ]

    return result


def explain_prediction(
    pipeline: Pipeline,
    text: str,
    target_names: List[str],
    top_k: int = 10,
    temperature: float = 1.0,
) -> Dict:
    """Explain one prediction: the topic, the probabilities, and the why.

    The "why" is computed on this document rather than read off the global
    weights - a token counts only if it actually appears. `temperature` applies
    the calibration from training, which cannot move the topic or the
    contributions. The runner-up class comes back too: the interesting cases
    are the close calls between sibling topics.
    """
    vectoriser = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    # Vectorise once; reuse the sparse row for both scoring and attribution.
    tfidf_row = vectoriser.transform([text])
    probabilities = (
        pipeline.predict_proba([text])[0]
        if temperature == 1.0
        else calibration.calibrated_probabilities(pipeline, [text], temperature)[0]
    )

    ranked = np.argsort(probabilities)[::-1]
    top_class_idx = int(ranked[0])
    runner_up_idx = int(ranked[1])

    contributions = _token_contributions(
        tfidf_row, classifier.coef_[top_class_idx],
        vectoriser.get_feature_names_out(), top_k,
    )

    return {
        "predicted_topic": target_names[top_class_idx],
        "confidence": float(probabilities[top_class_idx]),
        "runner_up_topic": target_names[runner_up_idx],
        "runner_up_confidence": float(probabilities[runner_up_idx]),
        "probabilities": {
            target_names[i]: float(probabilities[i]) for i in range(len(target_names))
        },
        "top_contributing_tokens": contributions,
    }


def _token_contributions(
    tfidf_row,
    class_weights: np.ndarray,
    feature_names: np.ndarray,
    top_k: int,
) -> List[Dict[str, float]]:
    """Rank the tokens present in one document by their signed contribution.

    Only the non-zero entries of the sparse row - the words that occur in the
    text. Contribution is tfidf times class weight, that word's additive term.
    """
    row = tfidf_row.tocoo()
    feature_names = np.asarray(feature_names)

    contributions = [
        {
            "token": str(feature_names[col]),
            "contribution": float(value * class_weights[col]),
        }
        for col, value in zip(row.col, row.data)
    ]

    # Most positive first: the words that pushed hardest toward the predicted topic.
    contributions.sort(key=lambda item: item["contribution"], reverse=True)
    return contributions[:top_k]
