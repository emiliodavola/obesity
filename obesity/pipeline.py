"""Classifier assembly and sklearn-compatible TabPFN model wrapper."""

from typing import Any, Self

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier


class TabPFNWrapper(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Expose TabPFN through sklearn estimator and feature-name contracts."""

    def __init__(self, estimator: Any) -> None:
        """Store the wrapped estimator."""
        self.estimator: Any = estimator

    def fit(self, X: Any, y: Any, **params: Any) -> Self:
        """Fit the estimator and retain any feature names available in X."""
        # 1. Attempt to capture feature names from the input (the truth)
        # In a pipeline, X is the output of the previous step.
        names: Any = None
        if hasattr(X, "columns"):
            names = list(X.columns)
        elif hasattr(X, "get_feature_names_out"):
            names = X.get_feature_names_out()

        self.feature_names_in_: Any = names if names is not None else []
        self.feature_names_out_: Any = names if names is not None else []

        self.estimator.fit(X, y, **params)
        self.classes_: Any = getattr(self.estimator, "classes_", None)
        return self

    def predict(self, X: Any) -> Any:
        """Predict target classes with the fitted estimator."""
        check_is_fitted(self)
        return self.estimator.predict(X)

    def transform(self, X: Any) -> Any:
        """Transform features with the fitted estimator."""
        check_is_fitted(self)
        return self.estimator.transform(X)

    def get_feature_names_out(self, feature_names_in: Any = None) -> Any:
        """Return supplied feature names or names captured during fitting."""
        check_is_fitted(self)
        # If the previous step provided names, use them (standard sklearn behavior)
        if feature_names_in is not None:
            return feature_names_in
        return self.feature_names_out_

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes not provided by the wrapper."""
        return getattr(self.estimator, name)


def create_pipeline(
    preprocessor: ColumnTransformer, model_path: str, device: str
) -> Pipeline:
    """Assemble the classifier pipeline."""
    model: TabPFNClassifier = TabPFNClassifier(model_path=model_path, device=device)
    # Wrap model to ensure sklearn/skore compliance for feature name propagation
    wrapped_model: TabPFNWrapper = TabPFNWrapper(model)
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", wrapped_model)]
    )
