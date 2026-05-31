from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from . import config


def build_tfidf():
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def stratified_split(texts, labels):
    texts = np.asarray(texts)
    labels = np.asarray(labels)

    # separamos test primero y no lo tocamos hasta el final
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts,
        labels,
        test_size=config.TEST_SIZE,
        stratify=labels,
        random_state=config.RANDOM_STATE,
    )

    # del 85% restante sacamos validacion para que sea 15% del total
    val_frac = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        stratify=y_trainval,
        random_state=config.RANDOM_STATE,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
