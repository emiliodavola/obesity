"""
The classifier assembly module.
"""

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier


class TabPFNWrapper(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **params):
        # 1. Attempt to capture feature names from the input (the truth)
        # In a pipeline, X is the output of the previous step.
        names = None
        if hasattr(X, 'columns'):
            names = list(X.columns)
        elif hasattr(X, 'get_feature_names_out'):
            names = X.get_feature_names_out()
        
        self.feature_names_in_ = names if names is not None else []
        self.feature_names_out_ = names if names is not None else []
        
        self.estimator.fit(X, y, **params)
        self.classes_ = getattr(self.estimator, 'classes_', None)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator.predict(X)

    def transform(self, X):
        check_is_fitted(self)
        return self.estimator.transform(X)

    def get_feature_names_out(self, feature_names_in=None):
        """Implements the sklearn contract for feature name propagation."""
        check_is_fitted(self)
        # If the previous step provided names, use them (standard sklearn behavior)
        if feature_names_in is not None:
            return feature_names_in
        return self.feature_names_out_

    def __getattr__(self, name):
        return getattr(self.estimator, name)


def create_pipeline(
    preprocessor: any,
    model_path: str,
    device: str
) -> Pipeline:
    """Assemble the classifier pipeline."""
    model = TabPFNClassifier(model_path=model_path, device=device)
    # Wrap model to ensure sklearn/skore compliance for feature name propagation
    wrapped_model = TabPFNWrapper(model)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", wrapped_model)])
