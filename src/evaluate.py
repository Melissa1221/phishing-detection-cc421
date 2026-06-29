from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def _proba_positive(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # si no hay predict_proba normalizamos el decision_function a [0,1]
    scores = model.decision_function(X)
    rng = float(np.ptp(scores))
    return (scores - scores.min()) / (rng + 1e-12)


def compute_metrics(model, X, y_true, name=""):
    y_pred = model.predict(X)
    y_score = _proba_positive(model, X)

    n_classes = len(np.unique(y_true))
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if n_classes > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if n_classes > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def meets_targets(metrics):
    return {
        "recall>=0.90": metrics["recall"] >= config.TARGET_RECALL,
        "f1>=0.88": metrics["f1"] >= config.TARGET_F1,
        "pr_auc>=0.92": metrics["pr_auc"] >= config.TARGET_PR_AUC,
    }


def save_confusion_matrix(model, X, y_true, name):
    cm = confusion_matrix(y_true, model.predict(X))
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusion - {name}")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"confusion_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_roc_pr_curves(models_scores, y_true):
    # ROC
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, y_score in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    roc_path = config.FIGURES_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, y_score in models_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    pr_path = config.FIGURES_DIR / "pr_curves.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    return str(roc_path), str(pr_path)


def save_metrics_json(all_metrics, filename="metrics.json"):
    path = config.METRICS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return str(path)


def threshold_sweep(model, X_val, y_val, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05).round(2).tolist()

    y_score = _proba_positive(model, X_val)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        })

    best = max(rows, key=lambda r: r["f1"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ts = [r["threshold"] for r in rows]
    ax.plot(ts, [r["precision"] for r in rows], "o-", label="Precision", color="#264653")
    ax.plot(ts, [r["recall"] for r in rows], "s-", label="Recall", color="#e76f51")
    ax.plot(ts, [r["f1"] for r in rows], "^-", label="F1", color="#2a9d8f")
    ax.axvline(best["threshold"], ls="--", alpha=0.5, color="#888")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold sweep (optimo F1={best['f1']:.4f} en t={best['threshold']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / "threshold_sweep.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    result = {"best_threshold": best, "sweep": rows}
    json_path = config.METRICS_DIR / "threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result

