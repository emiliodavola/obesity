"""Evaluation logic for the obesity classification pipeline."""

from typing import Any

from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_model(clf_pipeline: Any, X_test: Any, y_test: Any) -> tuple[float, float]:
    """Compute metrics for the model."""
    probs: Any = clf_pipeline.predict_proba(X_test)
    roc_auc: float = roc_auc_score(y_test, probs[:, 1])
    preds: Any = clf_pipeline.predict(X_test)
    acc: float = accuracy_score(y_test, preds)
    return roc_auc, acc
