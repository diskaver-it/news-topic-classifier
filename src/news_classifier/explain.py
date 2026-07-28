"""Interpretability: which words drove a prediction, and what defines a topic.

This is the part that makes a linear model worth choosing. For a TF-IDF +
logistic-regression pipeline the score for class c on a document is

    score_c = bias_c + sum over tokens t of  tfidf(t) * weight[c, t]

Every term of that sum is one word's signed contribution to one class. So we
can do two things no black-box model gives for free:

  * global - the highest-weight tokens for each class, i.e. the words the model
    treats as defining that topic;
  * local  - for a single document, the tokens that pushed it toward its
    predicted topic, ranked by their actual contribution on *this* text.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.pipeline import Pipeline

from . import config


def top_features_per_class(
    pipeline: Pipeline,
    target_names: List[str],
    top_k: int = config.TOP_FEATURES_PER_TOPIC,
) -> Dict[str, List[Dict[str, float]]]:
    """The tokens with the largest positive weight for each class.

    These are the model's learned "definition" of every topic, and they double
    as a sanity check: if sci.space's top words are launch, orbit, nasa,
    spacecraft, the model has learned the subject; if they were random function
    words, something is wrong with the vectoriser.
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
) -> Dict:
    """Explain one prediction: the topic, the probabilities, and the why.

    The "why" is computed on *this* document, not read off the global weights:
    a token only matters here if it actually appears (non-zero TF-IDF). The
    contribution of token t to the predicted class is tfidf(t) * weight[c, t],
    and ranking those surfaces the words that actually swung the decision.

    Returns a dict ready to hand straight to an API response or a UI, including
    the runner-up class - useful because the interesting cases are the close
    calls between sibling topics.
    """
    vectoriser = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    # Vectorise once; reuse the sparse row for both scoring and attribution.
    tfidf_row = vectoriser.transform([text])
    probabilities = pipeline.predict_proba([text])[0]

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

    Only the non-zero entries of the sparse row are considered - those are the
    words that actually occur in the text. For each, contribution = tf-idf value
    times the class weight, which is precisely that word's additive term in the
    linear score.
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
