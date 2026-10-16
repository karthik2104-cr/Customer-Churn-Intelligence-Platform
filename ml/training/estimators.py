"""Estimators that keep resampling and class weights inside fit."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted


class BalancedHistGradientBoosting(ClassifierMixin, BaseEstimator):
    """HistGradientBoosting with an optional balanced sample weight computed in fit."""

    def __init__(
        self,
        learning_rate: float = 0.08,
        max_depth: int | None = 6,
        max_iter: int = 200,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        class_weight: str | None = None,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        y_array = np.asarray(y)
        sample_weight = None
        if self.class_weight == "balanced":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_array)
        model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )
        model.fit(X, y_array, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


class RandomOversampleClassifier(ClassifierMixin, BaseEstimator):
    """Duplicate minority-class training rows inside fit. Does not touch validation rows."""

    def __init__(self, estimator=None, random_state: int = 42):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("An estimator is required.")
        y_array = np.asarray(y)
        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Oversampling requires both classes.")
        majority = int(counts.max())
        rng = np.random.default_rng(self.random_state)
        pieces = []
        for label, count in zip(classes, counts):
            label_index = np.flatnonzero(y_array == label)
            if int(count) < majority:
                extra = rng.choice(label_index, size=majority - int(count), replace=True)
                label_index = np.concatenate([label_index, extra])
            pieces.append(label_index)
        chosen = np.concatenate(pieces)
        rng.shuffle(chosen)
        if hasattr(X, "iloc"):
            x_fit = X.iloc[chosen]
        else:
            x_fit = X[chosen]
        estimator = clone(self.estimator)
        estimator.fit(x_fit, y_array[chosen])
        self.estimator_ = estimator
        self.classes_ = np.asarray(estimator.classes_)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)
