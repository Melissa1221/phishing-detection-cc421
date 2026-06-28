from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from . import config


# interpretar por pesos
def _extract_lr_coef(pipeline):
    """extrae (tfidf, coef_) de un Pipeline que tiene LogisticRegression como clf"""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
    # si es un GridSearchCV, el coef_ está en clf.estimator
    elif hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_"):
        coef = clf.estimator.coef_[0]
    # si es un Pipeline, el coef_ está en clf.steps[-1][1].coef_
    else:
        raise ValueError(f"clf {type(clf)} no expone coef_; usa LogisticRegression")
    return tfidf, coef


def save_top_weighted_terms(model, top_n: int = 30, name: str = "logistic_regression"):
    tfidf, coef = _extract_lr_coef(model)
    terms = tfidf.get_feature_names_out()

    idx_phishing = np.argsort(coef)[-top_n:][::-1]
    idx_legit = np.argsort(coef)[:top_n]

    result = {
        "top_phishing": [
            {"term": terms[i], "weight": float(coef[i])} for i in idx_phishing
        ],
        "top_legit": [
            {"term": terms[i], "weight": float(coef[i])} for i in idx_legit
        ],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ph = result["top_phishing"]
    ax1.barh([d["term"] for d in ph[::-1]], [d["weight"] for d in ph[::-1]], color="#e63946")
    ax1.set_title(f"Top {top_n} terminos → phishing")
    ax1.set_xlabel("Peso (coef)")

    lg = result["top_legit"]
    ax2.barh([d["term"] for d in lg[::-1]], [d["weight"] for d in lg[::-1]], color="#457b9d")
    ax2.set_title(f"Top {top_n} terminos → legitimo")
    ax2.set_xlabel("Peso (coef)")

    fig.suptitle(f"Pesos del modelo: {name}", fontsize=13)
    fig.tight_layout()

    png_path = config.FIGURES_DIR / "top_weighted_terms.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = config.METRICS_DIR / "top_weighted_terms.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[interpret] Pesos guardados en {png_path} y {json_path}")
    return result


# analizar errores de test: FN y FP
def error_analysis(model, X_test, y_test, source_test=None, texts_test=None):
    """devuelve FN y FP con texto, etiqueta real, probabilidad y source"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mask_fn = (y_test == 1) & (y_pred == 0)
    mask_fp = (y_test == 0) & (y_pred == 1)

    records = []
    for mask, etype in [(mask_fn, "FN"), (mask_fp, "FP")]:
        for i in np.where(mask)[0]:
            records.append({
                "error_type": etype,
                "label_real": int(y_test[i]),
                "prob_phishing": round(float(y_prob[i]), 4),
                "source": source_test[i] if source_test is not None else "desconocido",
                "text_snippet": texts_test[i][:200] if texts_test is not None else "",
            })

    df_err = pd.DataFrame(records)
    if df_err.empty:
        print("[interpret] Sin errores en test")
        return df_err

    resumen = df_err.groupby(["error_type", "source"]).size().reset_index(name="count")
    print("[interpret] Errores por fuente:")
    print(resumen.to_string(index=False))

    # longitud media y presencia de url_token por tipo de error
    df_err["longitud"] = df_err["text_snippet"].str.len()
    df_err["tiene_url_token"] = df_err["text_snippet"].str.contains(config.URL_TOKEN)

    csv_path = config.METRICS_DIR / "error_analysis.csv"
    df_err.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[interpret] Error analysis guardado en {csv_path}")
    return df_err


# analizar metricas por fuente
def metrics_by_source(model, X_test, y_test, source_test):
    y_pred = model.predict(X_test)
    sources = np.unique(source_test)

    rows = []
    for src in sources:
        mask = source_test == src
        if mask.sum() == 0:
            continue
        rows.append({
            "source": src,
            "n": int(mask.sum()),
            "precision": round(float(precision_score(y_test[mask], y_pred[mask], zero_division=0)), 4),
            "recall": round(float(recall_score(y_test[mask], y_pred[mask], zero_division=0)), 4),
            "f1": round(float(f1_score(y_test[mask], y_pred[mask], zero_division=0)), 4),
        })

    df_src = pd.DataFrame(rows)
    print("[interpret] Metricas por fuente:")
    print(df_src.to_string(index=False))

    json_path = config.METRICS_DIR / "metrics_by_source.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[interpret] Guardado en {json_path}")
    return df_src


# Tabla comparativa de metricas entre Enron-solo y multi-fuente
def comparative_table(metrics_enron: dict, metrics_multi: dict) -> pd.DataFrame:
    keys = ("precision", "recall", "f1", "roc_auc", "pr_auc")
    rows = [
        {
            "metrica": k,
            "enron_solo": round(metrics_enron[k], 4),
            "multi_fuente": round(metrics_multi[k], 4),
            "delta": round(metrics_multi[k] - metrics_enron[k], 4),
        }
        for k in keys
    ]
    df = pd.DataFrame(rows)
    print("[interpret] Tabla comparativa Enron-solo vs multi-fuente:")
    print(df.to_string(index=False))

    json_path = config.METRICS_DIR / "comparative_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"enron_solo": metrics_enron, "multi_fuente": metrics_multi, "delta": rows},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[interpret] Guardado en {json_path}")
    return df


# Auditoria adversarial
_ENRON_IDENTITY = re.compile(
    r"\b(enron|ect|hou|gas|energy)\b", re.IGNORECASE
)


def adversarial_audit(train_fn, X_train, y_train, X_test, y_test,
                      model_name: str = "logistic_regression"):
    model_base = train_fn(X_train, y_train)
    f1_base = f1_score(y_test, model_base.predict(X_test), zero_division=0)

    X_train_clean = np.array([_ENRON_IDENTITY.sub("", t) for t in X_train])
    X_test_clean = np.array([_ENRON_IDENTITY.sub("", t) for t in X_test])

    model_clean = train_fn(X_train_clean, y_train)
    f1_clean = f1_score(y_test, model_clean.predict(X_test_clean), zero_division=0)

    delta = f1_clean - f1_base
    result = {
        "model": model_name,
        "identity_tokens_removed": ["enron", "ect", "hou", "gas", "energy"],
        "f1_con_tokens": round(float(f1_base), 4),
        "f1_sin_tokens": round(float(f1_clean), 4),
        "delta_f1": round(float(delta), 4),
    }
    print(
        f"[interpret] Auditoria adversarial: F1 base={f1_base:.4f}, "
        f"F1 sin tokens={f1_clean:.4f}, delta_f1={delta:+.4f}"
    )

    json_path = config.METRICS_DIR / "adversarial_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[interpret] Guardado en {json_path}")
    return result
