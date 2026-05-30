"""Classical scikit-learn models: random forest and a logistic-regression baseline."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .. import config


class RandomForestModel:
    """Thin wrapper around :class:`sklearn.ensemble.RandomForestClassifier`."""

    is_deep = False

    def __init__(self, cfg: config.RandomForestConfig, seed: int) -> None:
        self.cfg = cfg
        self.history: dict | None = None
        self.clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight=cfg.class_weight,
            n_jobs=cfg.n_jobs,
            random_state=seed,
        )

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_

    def complexity(self) -> dict:
        n_nodes = sum(t.tree_.node_count for t in self.clf.estimators_)
        return {"n_estimators": self.cfg.n_estimators, "total_tree_nodes": int(n_nodes)}


class LogisticRegressionModel:
    """Linear baseline so we can quantify how much non-linearity buys us."""

    is_deep = False

    def __init__(self, seed: int) -> None:
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        self.history = None

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def complexity(self) -> dict:
        return {"n_coefficients": int(self.clf.coef_.size + self.clf.intercept_.size)}
