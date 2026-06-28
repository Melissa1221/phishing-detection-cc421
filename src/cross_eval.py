from __future__ import annotations

import json
import logging

from . import config
from .data_loader import load_corpus
from .evaluate import compute_metrics
from .preprocess import preprocess_series
from .train import build_models

log = logging.getLogger(__name__)

_LOCO_EXPERIMENTS = [
    {"train": ["enron", "spamassassin"], "test": "nazario"},
    {"train": ["enron", "nazario"], "test": "spamassassin"},
    {"train": ["spamassassin", "nazario"], "test": "enron"},
]


def run_cross_eval():
    """Leave-one-corpus-out completo: 3 experimentos."""
    df = load_corpus()
    df["clean_text"] = preprocess_series(df["text"])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    results = []
    for exp in _LOCO_EXPERIMENTS:
        train_sources = exp["train"]
        test_source = exp["test"]

        train_mask = df["source"].isin(train_sources)
        test_mask = df["source"] == test_source

        X_train = df.loc[train_mask, "clean_text"].values
        y_train = df.loc[train_mask, "label"].values
        X_test = df.loc[test_mask, "clean_text"].values
        y_test = df.loc[test_mask, "label"].values

        if len(X_test) == 0:
            log.warning("Sin datos de test para source=%s, se omite.", test_source)
            continue

        label_name = f"lr_{'+'.join(train_sources)}_to_{test_source}"
        log.info(
            "LOCO train=%d (%s), test=%d (%s)",
            len(X_train), "+".join(train_sources), len(X_test), test_source,
        )

        model = build_models()["logistic_regression"]
        model.fit(X_train, y_train)
        m = compute_metrics(model, X_test, y_test, label_name)
        m["train_sources"] = train_sources
        m["test_source"] = test_source
        results.append(m)

        log.info(
            "LOCO %s → F1=%.4f Recall=%.4f Precision=%.4f",
            label_name, m["f1"], m["recall"], m["precision"],
        )

    out = config.METRICS_DIR / "cross_corpus_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Cross-corpus eval guardado en %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cross_eval()
