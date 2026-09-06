"""Training entry point.

Run with:  python -m news_classifier.train

Tune C by cross-validation and score on the official holdout, then the two
stages that lift this above a fit-predict script: calibrate the confidence and
derive the point at which the model should refuse to answer, and measure how
many points of "accuracy" the headers, footers and quotes hand over for free.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from . import calibration, config, data, evaluate, explain, model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
)
logger = logging.getLogger(__name__)


def tune_regularisation(X_train, y_train) -> float:
    """Pick the logistic-regression C by 3-fold CV on macro-F1.

    Macro-F1 rather than accuracy so the choice stays honest about the harder,
    smaller classes instead of being decided by the big easy ones.
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


def calibrate_and_choose_abstention(pipeline, ds, best_c: float) -> Dict:
    """Make the confidence mean something, then use it to decide when to refuse.

    Two rules, and they are why this is a stage rather than three lines
    appended to the evaluation. The temperature is fitted out of fold -
    in-sample logits are the model's opinion of answers it has already seen.
    And the abstention threshold is chosen on the training folds and only
    measured on the holdout: picking the cutoff that hits 90% on the test set
    and then reporting 90% on the test set is circular.
    """
    logger.info("Fitting temperature on out-of-fold logits (%d folds) ...",
                config.CALIBRATION_CV_FOLDS)

    oof_logits = cross_val_predict(
        model.build_pipeline(C=best_c),
        ds.X_train,
        ds.y_train,
        cv=StratifiedKFold(
            n_splits=config.CALIBRATION_CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        ),
        method="decision_function",
        n_jobs=-1,
    )
    temperature = calibration.fit_temperature(oof_logits, ds.y_train)
    direction = ("sharpens - the model was under-confident" if temperature < 1
                 else "softens - the model was over-confident")
    logger.info("  temperature = %.4f  (%s)", temperature, direction)

    test_logits = pipeline.decision_function(ds.X_test)
    report = calibration.evaluate_calibration(test_logits, ds.y_test, temperature)
    logger.info("  holdout ECE  %.4f -> %.4f   (accuracy unchanged: %s)",
                report["uncalibrated"]["ece"], report["calibrated"]["ece"],
                report["accuracy_unchanged"])
    logger.info("  mean confidence %.4f vs accuracy %.4f before scaling",
                report["uncalibrated"]["mean_confidence"],
                report["uncalibrated"]["accuracy"])

    # Choose the cutoff on the training folds ...
    oof_probabilities = calibration.softmax(oof_logits, temperature)
    oof_confidence = oof_probabilities.max(axis=1)
    oof_correct = oof_probabilities.argmax(axis=1) == np.asarray(ds.y_train)
    operating_point = calibration.threshold_for_target_accuracy(
        oof_confidence, oof_correct, config.ABSTAIN_TARGET_ACCURACY
    )

    # ... and find out what it does on documents nobody tuned against.
    test_probabilities = calibration.softmax(test_logits, temperature)
    test_confidence = test_probabilities.max(axis=1)
    test_correct = test_probabilities.argmax(axis=1) == np.asarray(ds.y_test)

    holdout: Dict[str, float] = {}
    if operating_point["achievable"]:
        threshold = operating_point["threshold"]
        answered = test_confidence >= threshold
        holdout = {
            "threshold": threshold,
            "coverage": float(answered.mean()),
            "n_answered": int(answered.sum()),
            "selective_accuracy": (
                float(test_correct[answered].mean()) if answered.any() else 0.0
            ),
            "accuracy_if_answering_everything": float(test_correct.mean()),
        }
        logger.info(
            "  abstain below %.3f: answers %.1f%% of the holdout at %.4f accuracy "
            "(vs %.4f answering everything)",
            threshold, 100 * holdout["coverage"], holdout["selective_accuracy"],
            holdout["accuracy_if_answering_everything"],
        )
    else:
        logger.info("  no usable abstention threshold: %s", operating_point["note"])

    return {
        "temperature": temperature,
        "calibration": report,
        "operating_point_chosen_on_training_folds": operating_point,
        "operating_point_measured_on_holdout": holdout,
        "risk_coverage_curve": calibration.risk_coverage_curve(
            test_confidence, test_correct
        ),
        "reliability_bins": {
            "uncalibrated": calibration.reliability_bins(
                calibration.softmax(test_logits, 1.0).max(axis=1), test_correct
            ),
            "calibrated": calibration.reliability_bins(test_confidence, test_correct),
        },
    }


def measure_leakage_effect() -> Dict:
    """Quantify what the headers, footers and quotes are worth - and why we cut them.

    The 20 Newsgroups analogue of a target leak: the posts carry their
    newsgroup in the headers and on-topic quoted text in the body, so a model
    can score well while reading metadata rather than language. Train it both
    ways, report both, keep the lower one.
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

    # ---- 3. Confidence: calibration and abstention ----------------------
    logger.info("-" * 74)
    logger.info("Confidence - is a '99%%' worth believing, and when to refuse?")
    confidence = calibrate_and_choose_abstention(pipeline, ds, best_c)

    # ---- 4. Interpretability -------------------------------------------
    top_features = explain.top_features_per_class(pipeline, ds.target_names)
    logger.info("-" * 74)
    logger.info("Sanity check - learned defining words:")
    for topic in ("sci.space", "rec.sport.hockey", "sci.crypt"):
        words = ", ".join(f["token"] for f in top_features[topic][:6])
        logger.info("  %-20s %s", topic, words)

    # ---- 5. Leakage experiment -----------------------------------------
    logger.info("-" * 74)
    logger.info("Leakage experiment - what the post metadata would buy us:")
    leakage = measure_leakage_effect()

    # ---- 6. Persist ----------------------------------------------------
    import joblib

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "target_names": ds.target_names,
        "best_C": best_c,
        # Both travel with the model so the demo cannot drift from the numbers
        # in reports/: one scalar to calibrate the confidence, one cutoff below
        # which the honest answer is "not sure".
        "temperature": confidence["temperature"],
        "abstain_threshold": confidence[
            "operating_point_chosen_on_training_folds"
        ].get("threshold"),
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
        "confidence": {
            "temperature": confidence["temperature"],
            "ece_before": confidence["calibration"]["uncalibrated"]["ece"],
            "ece_after": confidence["calibration"]["calibrated"]["ece"],
            "abstention": confidence["operating_point_measured_on_holdout"],
        },
        "environment": {"sklearn": sklearn.__version__, "python": platform.python_version()},
    }
    config.METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    config.LEAKAGE_REPORT_FILE.write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    config.TOP_FEATURES_FILE.write_text(json.dumps(top_features, indent=2), encoding="utf-8")
    config.CALIBRATION_REPORT_FILE.write_text(
        json.dumps(confidence, indent=2), encoding="utf-8"
    )
    logger.info("Saved reports -> %s", config.REPORTS_DIR.name)

    # ---- 7. Figures ----------------------------------------------------
    try:
        from .plots import make_all_figures

        make_all_figures(
            ds.y_test, y_pred, ds.target_names, metrics["per_class"], top_features,
            confidence=confidence,
        )
        logger.info("Saved figures -> %s", config.FIGURES_DIR)
    except Exception as exc:            # plotting must never break training
        logger.warning("Figures skipped: %s", exc)

    logger.info("=" * 74)
    logger.info("Done.")
    return report


if __name__ == "__main__":
    main()
