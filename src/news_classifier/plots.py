"""Report figures. Kept separate so training never depends on plotting."""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")   # headless: writes files, needs no display (CI, SSH)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config, evaluate  # noqa: E402


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], target_names: List[str]
) -> None:
    """Row-normalised 20x20 heatmap - the whole error structure in one picture.

    A bright diagonal means the model is right; off-diagonal bright spots are
    the systematic confusions, and they cluster into blocks that line up with
    the super-categories (the comp.* corner, the talk.*/religion corner).
    """
    matrix = evaluate.confusion_matrix_normalised(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=90, fontsize=7)
    ax.set_yticklabels(target_names, fontsize=7)
    ax.set_xlabel("Predicted topic")
    ax.set_ylabel("True topic")
    ax.set_title("Confusion matrix (row-normalised, holdout)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    _save(fig, "confusion_matrix.png")


def plot_per_class_f1(per_class: Dict, target_names: List[str]) -> None:
    """Per-topic F1 as a sorted bar chart - which topics are easy, which hard.

    Sorting makes the ranking obvious at a glance: the clean subject-matter
    groups (sci.space, rec.sport.hockey) sit at the top, the vaguer ones
    (talk.politics.misc, talk.religion.misc) at the bottom.
    """
    scores = [(name, per_class[name]["f1-score"]) for name in target_names]
    scores.sort(key=lambda item: item[1])

    names = [s[0] for s in scores]
    values = [s[1] for s in scores]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.RdYlGn(np.asarray(values))
    ax.barh(names, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-topic F1 (holdout)")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    _save(fig, "per_class_f1.png")


def plot_top_features(top_features: Dict[str, List[Dict[str, float]]],
                      topics: List[str], top_k: int = 10) -> None:
    """Small-multiples of the defining words for a handful of topics.

    Not all 20 - a 4-topic sample is enough to show that the learned weights are
    sensible subject-matter words, which is the point of the figure.
    """
    fig, axes = plt.subplots(1, len(topics), figsize=(4 * len(topics), 5))
    if len(topics) == 1:
        axes = [axes]

    for ax, topic in zip(axes, topics):
        items = top_features[topic][:top_k][::-1]
        tokens = [it["token"] for it in items]
        weights = [it["weight"] for it in items]
        ax.barh(tokens, weights, color="steelblue")
        ax.set_title(topic, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Top weighted tokens per topic", y=1.02)
    _save(fig, "top_features.png")


def make_all_figures(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
    per_class: Dict,
    top_features: Dict[str, List[Dict[str, float]]],
) -> None:
    plot_confusion_matrix(y_true, y_pred, target_names)
    plot_per_class_f1(per_class, target_names)
    # A representative spread across super-categories.
    sample = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
    plot_top_features(top_features, [t for t in sample if t in top_features])
