from __future__ import annotations

import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config
from .features import build_tfidf


def build_models():
    # cada modelo es un pipeline tfidf -> clf para poder serializar todo junto
    rs = config.RANDOM_STATE
    models = {
        "naive_bayes": Pipeline(
            [("tfidf", build_tfidf()), ("clf", MultinomialNB())]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=rs,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=rs,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # LinearSVC no da predict_proba, lo calibramos con Platt para tener probabilidades
        "svm_linear": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            random_state=rs,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }
    return models


def train_model(name, model, X_train, y_train):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"[train] {name}: entrenado en {elapsed:.2f}s")
    return model, elapsed


def save_model(name, model):
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path)
