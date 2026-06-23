"""Evaluation logic for the obesity classification pipeline."""

from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_model(clf_pipeline: any, X_test: any, y_test: any) -> tuple[float, float]:
    """Compute metrics for the model."""
    probs = clf_pipeline.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probs[:, 1])
    preds = clf_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return roc_auc, acc
