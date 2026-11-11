"""Train, tune and compare churn models, then save the best one.

Run with:  python -m churn.train            (full grid)
           python -m churn.train --fast     (small grid, for a quick check)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from churn.config import (
    FIGURES_DIR, METRICS_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE, RAW_FEATURES, REPORTS_DIR, TEST_SIZE,
)
from churn.data import load_data
from churn.features import FeatureEngineer, build_preprocessor
from churn.nn_model import TorchMLPClassifier


def make_pipeline(model) -> Pipeline:
    """features -> scale/encode -> SMOTE (train folds only) -> model."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model),
    ])


def model_grid(fast: bool = False) -> dict[str, tuple[object, dict, int]]:
    """name -> (estimator, param grid, n_jobs)."""
    if fast:
        return {
            "Logistic Regression": (LogisticRegression(max_iter=2000), {"model__C": [1.0]}, -1),
            "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                              {"model__n_estimators": [100], "model__max_depth": [8]}, 1),
            "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
                        {"model__n_estimators": [100], "model__max_depth": [3]}, 1),
            "PyTorch MLP": (TorchMLPClassifier(epochs=20), {"model__lr": [1e-3]}, 1),
        }
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": [0.01, 0.1, 1.0, 10.0]}, -1),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [300], "model__max_depth": [6, 10, None],
             "model__min_samples_leaf": [1, 5]}, 1),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5],
             "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8]}, 1),
        "PyTorch MLP": (
            TorchMLPClassifier(),
            {"model__hidden_dims": [(64, 32), (128, 64)], "model__lr": [1e-3, 3e-3],
             "model__dropout": [0.3]}, 1),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Decision threshold that maximises F1 (instead of the default 0.5)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "threshold": round(threshold, 3),
    }


def main(fast: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | churn rate {y.mean():.1%}")

    cv = StratifiedKFold(n_splits=3 if fast else 5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, (estimator, grid, n_jobs) in model_grid(fast).items():
        start = time.time()
        search = GridSearchCV(make_pipeline(estimator), grid, scoring="roc_auc", cv=cv,
                              n_jobs=n_jobs, refit=True)
        search.fit(X_train, y_train)
        proba = search.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba, 0.5)
        metrics["cv_roc_auc"] = round(search.best_score_, 4)
        metrics["best_params"] = {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
                                  for k, v in search.best_params_.items()}
        metrics["train_seconds"] = round(time.time() - start, 1)
        results[name], fitted[name] = metrics, search.best_estimator_
        print(f"{name:20s} CV AUC {metrics['cv_roc_auc']:.4f} | test AUC {metrics['roc_auc']:.4f} "
              f"| F1@0.5 {metrics['f1']:.4f} | {metrics['train_seconds']}s")

    # pick the model with the best cross-validated ROC-AUC (never peek at the test set to choose)
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best = fitted[best_name]

    # tune the decision threshold on out-of-fold predictions from the training set
    oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    best_proba = best.predict_proba(X_test)[:, 1]
    best_metrics = evaluate(y_test, best_proba, threshold)
    print(f"\nBest model: {best_name} | tuned threshold {threshold:.3f} -> test {best_metrics}")

    # global feature importance on the raw (business-readable) columns
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=1)
    importance = (pd.Series(imp.importances_mean, index=X_test.columns)
                  .sort_values(ascending=False).round(4))

    # ---- save artefacts ------------------------------------------------------
    joblib.dump({
        "pipeline": best,
        "model_name": best_name,
        "threshold": threshold,
        "features": RAW_FEATURES,
        "background": X_train.sample(200, random_state=RANDOM_STATE).reset_index(drop=True),
    }, MODEL_PATH)

    report = {
        "best_model": best_name,
        "best_model_test_metrics": best_metrics,
        "all_models": results,
        "feature_importance": importance.to_dict(),
        "n_train": len(X_train), "n_test": len(X_test), "churn_rate": round(float(y.mean()), 4),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves (test set)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, (best_proba >= threshold).astype(int),
                                            display_labels=["Stay", "Churn"], ax=ax, colorbar=False)
    ax.set_title(f"{best_name} @ threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    importance.head(12)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("Top churn drivers (permutation importance)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120); plt.close(fig)

    print(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="tiny grid for a quick smoke test")
    main(fast=parser.parse_args().fast)
