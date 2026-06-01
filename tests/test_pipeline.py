from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.features import build_tfidf, stratified_split
from src.preprocess import clean_text
from src.train import build_models


def test_clean_text_replaces_special_tokens():
    out = clean_text("Visit http://x.com email a@b.com call 123-456")
    assert config.URL_TOKEN in out
    assert config.EMAIL_TOKEN in out
    assert config.NUM_TOKEN in out
    assert not any(ch.isdigit() for ch in out)


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_stratified_split_proportions():
    texts = np.array([f"correo {i} phishing money" if i % 3 == 0
                      else f"correo {i} reunion agenda" for i in range(300)])
    labels = np.array([1 if i % 3 == 0 else 0 for i in range(300)])
    s = stratified_split(texts, labels)
    n = len(texts)
    assert abs(len(s["X_test"]) / n - config.TEST_SIZE) < 0.02
    assert abs(len(s["X_val"]) / n - config.VAL_SIZE) < 0.02
    # la proporcion de clase positiva se mantiene en cada particion
    base = labels.mean()
    for part in ("y_train", "y_val", "y_test"):
        assert abs(s[part].mean() - base) < 0.05


def test_models_train_and_predict():
    X = [
        "urgent verify account money url_token click now",
        "free prize winner claim money url_token",
        "meeting agenda quarterly report attached",
        "lunch tomorrow noon conference room",
    ] * 8
    y = [1, 1, 0, 0] * 8
    for name, model in build_models().items():
        model.fit(X, y)
        proba = model.predict_proba(X[:2])
        assert proba.shape == (2, 2)
        assert np.all((proba >= 0) & (proba <= 1))


def test_determinism():
    # mismo seed -> mismo split
    texts = np.array([f"t{i}" for i in range(100)])
    labels = np.array([i % 2 for i in range(100)])
    a = stratified_split(texts, labels)
    b = stratified_split(texts, labels)
    assert np.array_equal(a["X_train"], b["X_train"])
    assert np.array_equal(a["y_test"], b["y_test"])


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print(f"  ok {fn.__name__}")
    print("Todas las pruebas pasaron.")
