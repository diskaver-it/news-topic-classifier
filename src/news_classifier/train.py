"""Training entry point.

Run with:  python -m news_classifier.train

Three stages, and the last two are what lift this above a fit-predict script:

  1. tune the regularisation strength by cross-validation, then fit and score
     on the official holdout;
  2. the leakage experiment - retrain with the post headers/footers/quotes left
     in and measure how many points of "accuracy" they hand over for free;
  3. persist the model, the per-topic defining words, and the report figures.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config, data, evaluate, explain, model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
)
logger = logging.getLogger(__name__)


def tune_regularisation(X_train, y_train) -> float:
    """Pick the logistic-regression C by 3-fold CV on macro-F1.

    A small, explicit grid rather than a blind default: too much regularisation
    (small C) underfits 20 fine-grained topics, too little overfits the sparse
    tail of the vocabulary. Optimising macro-F1 rather than accuracy keeps the
    choice honest about the harder, smaller-signal classes.
    """
    logger.info("Tuning C by cross-validation ...")

    grid = GridSearchCV(
        estimator=model.build_pipeline(),
        param_grid={"clf__C": [0.3, 1.0, 3.0, 10.0]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    for c, score in zip(
        grid.cv_results_["param_clf__C"], grid.cv_results_["mean_test_score"]
    ):
        logger.info("    C=%-5s  macro-F1 = %.4f", c, score)
    logger.info("Best C = %s (CV macro-F1 %.4f)", grid.best_params_["clf__C"],
                grid.best_score_)

    return float(grid.best_params_["clf__C"])


def measure_leakage_effect() -> Dict:
    """Quantify what the post headers/footers/quotes are worth - and why we cut them.

    This is the 20 Newsgroups analogue of a target leak. The raw posts carry
    their newsgroup in the headers, recurring signatures in the footers, and
    quoted parent text that is usually on-topic. A model trained on all that can
    hit very high accuracy while doing almost no language understanding - it
    reads the metadata. Strip those parts and the score drops to what the model
    can actually earn from the message body.

    We train the same pipeline both ways and report both numbers. The honest one
    is the lower one.
    """
    results: Dict[str, Dict] = {}

    for label, remove in [("content_only", config.REMOVE_PARTS), ("with_metadata", ())]:
        ds = data.load_dataset(remove=remove)
        pipeline = model.build_pipeline()
        pipeline.fit(ds.X_train, ds.y_train)
        y_pred = pipeline.predict(ds.X_test)

        metrics = evaluate.evaluate_predictions(ds.y_test, y_pred, ds.target_names)
        results[label] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        logger.info(
            "  %-14s accuracy=%.4f  macro-F1=%.4f",
            label, metrics["accuracy"], metrics["macro_f1"],
        )

    content, meta = results["content_only"], results["with_metadata"]
    results["inflation"] = {
        "accuracy_points": round(meta["accuracy"] - content["accuracy"], 4),
        "macro_f1_points": round(meta["macro_f1"] - content["macro_f1"], 4),
        "note": (
            "Leaving the headers, footers and quoted text in inflates the score "
            "by this much. Those parts often name the newsgroup outright, so the "
            "model learns to read metadata instead of understanding the message. "
            "The shipped model is trained on message content only."
        ),
    }
    return results


def main() -> Dict:
    logger.info("=" * 74)
    logger.info("20 Newsgroups - topic classifier")
    logger.info("=" * 74)

    # ---- 1. Data -------------------------------------------------------
    ds = data.load_dataset()

    # ---- 2. Tune, fit, evaluate ----------------------------------------
    logger.info("-" * 74)
    best_c = tune_regularisation(ds.X_train, ds.y_train)

    logger.info("Fitting final model on the full training split ...")
    pipeline = model.build_pipeline(C=best_c)
    pipeline.fit(ds.X_train, ds.y_train)

    y_pred = pipeline.predict(ds.X_test)
    metrics = evaluate.evaluate_predictions(ds.y_test, y_pred, ds.target_names)
    confusions = evaluate.top_confusions(ds.y_test, y_pred, ds.target_names)

    logger.info("-" * 74)
    logger.info("Holdout (official by-date test split):")
    logger.info("  accuracy     %.4f", metrics["accuracy"])
    logger.info("  macro-F1     %.4f", metrics["macro_f1"])
    logger.info("  weighted-F1  %.4f", metrics["weighted_f1"])

    same = sum(c["same_supercategory"] for c in confusions)
    logger.info("  of the top %d confusions, %d are between sibling topics "
                "(same super-category)", len(confusions), same)
    for c in confusions[:5]:
        logger.info("    %-26s -> %-26s  %4d%s", c["true"], c["predicted"],
                    c["count"], "  (siblings)" if c["same_supercategory"] else "")

    # ---- 3. Interpretability -------------------------------------------
    top_features = explain.top_features_per_class(pipeline, ds.target_names)
    logger.info("-" * 74)
    logger.info("Sanity check - learned defining words:")
    for topic in ("sci.space", "rec.sport.hockey", "sci.crypt"):
        words = ", ".join(f["token"] for f in top_features[topic][:6])
        logger.info("  %-20s %s", topic, words)

    # ---- 4. Leakage experiment -----------------------------------------
    logger.info("-" * 74)
    logger.info("Leakage experiment - what the post metadata would buy us:")
    leakage = measure_leakage_effect()

    # ---- 5. Persist ----------------------------------------------------
    import joblib

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "target_names": ds.target_names,
        "best_C": best_c,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_train_docs": len(ds.X_train),
    }
    joblib.dump(artifact, config.MODEL_FILE)
    logger.info("-" * 74)
    logger.info("Saved model artifact -> %s", config.MODEL_FILE.name)

    report = {
        "trained_at": artifact["trained_at"],
        "dataset": {
            "train_docs": len(ds.X_train),
            "test_docs": len(ds.X_test),
            "n_classes": ds.n_classes,
            "split": "official 20 Newsgroups by-date train/test",
            "removed_parts": list(config.REMOVE_PARTS),
        },
        "best_C": best_c,
        "holdout_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": metrics["per_class"],
        },
        "top_confusions": confusions,
        "environment": {"sklearn": sklearn.__version__, "python": platform.python_version()},
    }
    config.METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    config.LEAKAGE_REPORT_FILE.write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    config.TOP_FEATURES_FILE.write_text(json.dumps(top_features, indent=2), encoding="utf-8")
    logger.info("Saved reports -> %s", config.REPORTS_DIR.name)

    # ---- 6. Figures ----------------------------------------------------
    try:
        from .plots import make_all_figures

        make_all_figures(
            ds.y_test, y_pred, ds.target_names, metrics["per_class"], top_features
        )
        logger.info("Saved figures -> %s", config.FIGURES_DIR)
    except Exception as exc:            # plotting must never break training
        logger.warning("Figures skipped: %s", exc)

    logger.info("=" * 74)
    logger.info("Done.")
    return report


if __name__ == "__main__":
    main()
