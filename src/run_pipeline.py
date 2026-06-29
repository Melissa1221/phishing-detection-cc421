from __future__ import annotations

import argparse
import json
import logging
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

from . import config, eda, interpret
from .data_loader import load_corpus, load_enron
from .evaluate import (
    _proba_positive,
    compute_metrics,
    meets_targets,
    save_confusion_matrix,
    save_metrics_json,
    save_roc_pr_curves,
    threshold_sweep,
)
from .features import stratified_split
from .preprocess import preprocess_series
from .train import build_models, save_model, train_model, tune_model

log = logging.getLogger("pipeline")


def _setup_logging():
    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")


def set_seeds():
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)


def save_learning_curve(name, model, X, y):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", color="#264653", label="F1 entrenamiento")
    ax.plot(sizes, val_scores.mean(axis=1), "o-", color="#e76f51", label="F1 validacion (CV)")
    ax.fill_between(
        sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
        color="#e76f51",
    )
    ax.set_xlabel("Tamano del conjunto de entrenamiento")
    ax.set_ylabel("F1-score")
    ax.set_title(f"Curva de aprendizaje - {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"learning_curve_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Usar solo N correos (debug).")
    parser.add_argument("--no-cv", action="store_true", help="Saltar validacion cruzada.")
    args = parser.parse_args(argv)

    _setup_logging()
    set_seeds()
    t_start = time.perf_counter()

    log.info("=== 1) Carga y consolidacion del corpus ===")
    df = load_corpus()
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=config.RANDOM_STATE)
        df = df.reset_index(drop=True)
        log.info("Muestreo de depuracion: %d correos.", len(df))

    log.info("=== 2) Limpieza y normalizacion ===")
    t0 = time.perf_counter()
    df["clean_text"] = preprocess_series(df["text"])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    log.info("Limpieza completada en %.1fs (%d correos no vacios).", time.perf_counter() - t0, len(df))
    df.to_csv(config.PROCESSED_DIR / "corpus_clean.csv", index=False)

    log.info("=== 3) Analisis exploratorio (EDA) ===")
    eda_summary = eda.run_eda(df, clean_col="clean_text")
    log.info("Distribucion: %s", json.dumps(eda_summary["class_distribution"], ensure_ascii=False))

    log.info("=== 4) Particion estratificada 70/15/15 ===")
    split = stratified_split(
        df["clean_text"].values, df["label"].values, source=df["source"].values
    )
    log.info(
        "train=%d, val=%d, test=%d",
        len(split["X_train"]), len(split["X_val"]), len(split["X_test"]),
    )

    log.info("=== 5) Entrenamiento y ajuste de hiperparametros (GridSearchCV) ===")
    models = build_models()
    trained = {}
    train_times = {}
    cv_results = {}
    best_params: dict[str, dict] = {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, model in models.items():
        fitted, elapsed = tune_model(name, model, split["X_train"], split["y_train"])
        trained[name] = fitted
        train_times[name] = elapsed
        if hasattr(fitted, "best_params_"):
            best_params[name] = fitted.best_params_
        if not args.no_cv:
            scores = cross_val_score(
                build_models()[name],
                split["X_train"],
                split["y_train"],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )
            cv_results[name] = {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}
            log.info("[cv] %s: F1 = %.4f +/- %.4f", name, scores.mean(), scores.std())

    if best_params:
        bp_path = config.METRICS_DIR / "best_hyperparams.json"
        with open(bp_path, "w", encoding="utf-8") as _f:
            json.dump(best_params, _f, indent=2, ensure_ascii=False)
        log.info("Hiperparametros optimos guardados en %s", bp_path)

    log.info("=== 6) Evaluacion en validacion ===")
    val_metrics = []
    val_scores_for_curves = {}
    for name, model in trained.items():
        m = compute_metrics(model, split["X_val"], split["y_val"], name)
        m["train_seconds"] = round(train_times[name], 3)
        if name in cv_results:
            m["cv_f1_mean"] = cv_results[name]["f1_mean"]
            m["cv_f1_std"] = cv_results[name]["f1_std"]
        val_metrics.append(m)
        val_scores_for_curves[name] = _proba_positive(model, split["X_val"])
        log.info(
            "[val] %s: F1=%.4f recall=%.4f precision=%.4f PR-AUC=%.4f",
            name, m["f1"], m["recall"], m["precision"], m["pr_auc"],
        )

    best_name = max(val_metrics, key=lambda d: d["f1"])["model"]
    best_model = trained[best_name]
    log.info("Mejor modelo por F1 en validacion: %s", best_name)

    save_roc_pr_curves(val_scores_for_curves, split["y_val"])
    for name, model in trained.items():
        save_confusion_matrix(model, split["X_val"], split["y_val"], name)
    save_metrics_json(val_metrics, "val_metrics.json")

    log.info("=== 6b) Threshold sweep en validacion (mejor modelo) ===")
    sweep = threshold_sweep(best_model, split["X_val"], split["y_val"])
    log.info(
        "Threshold optimo: t=%.2f  F1=%.4f",
        sweep["best_threshold"]["threshold"], sweep["best_threshold"]["f1"],
    )

    log.info("=== 7) Evaluacion final en TEST (mejor modelo) ===")
    test_metrics = compute_metrics(best_model, split["X_test"], split["y_test"], best_name)
    targets = meets_targets(test_metrics)
    test_metrics["targets_met"] = targets
    save_confusion_matrix(best_model, split["X_test"], split["y_test"], f"{best_name}_TEST")
    save_metrics_json([test_metrics], "test_metrics.json")
    log.info(
        "TEST %s: F1=%.4f recall=%.4f precision=%.4f PR-AUC=%.4f",
        best_name, test_metrics["f1"], test_metrics["recall"],
        test_metrics["precision"], test_metrics["pr_auc"],
    )

    log.info("=== 8) Curva de aprendizaje (modelo ganador) ===")
    save_learning_curve(best_name, build_models()[best_name], split["X_train"], split["y_train"])

    log.info("=== 9) Interpretabilidad por pesos del modelo ===")
    if "logistic_regression" in trained:
        interpret.save_top_weighted_terms(trained["logistic_regression"])
    else:
        log.warning("logistic_regression no entrenado; se omite analisis de pesos.")
    if "random_forest" in trained:
        interpret.save_rf_feature_importances(trained["random_forest"])
    else:
        log.warning("random_forest no entrenado; se omite feature importances.")

    log.info("=== 10) Analisis de errores ===")
    source_test = split.get("source_test")
    interpret.error_analysis(
        best_model,
        split["X_test"],
        split["y_test"],
        source_test=source_test,
        texts_test=split["X_test"],
    )
    if source_test is not None:
        interpret.metrics_by_source(best_model, split["X_test"], split["y_test"], source_test)

    log.info("=== 11) Tabla comparativa Enron-solo vs multi-fuente ===")
    df_enron = load_enron()
    df_enron["text"] = (df_enron["subject"] + " " + df_enron["message"]).str.strip()
    df_enron["clean_text"] = preprocess_series(df_enron["text"])
    df_enron = df_enron[df_enron["clean_text"].str.len() > 0].reset_index(drop=True)
    if args.sample:
        df_enron = df_enron.sample(
            n=min(args.sample, len(df_enron)), random_state=config.RANDOM_STATE
        ).reset_index(drop=True)
    split_enron = stratified_split(df_enron["clean_text"].values, df_enron["label"].values)
    model_enron_lr = build_models()["logistic_regression"]
    train_model("logistic_regression_enron", model_enron_lr, split_enron["X_train"], split_enron["y_train"])
    m_enron = compute_metrics(model_enron_lr, split_enron["X_test"], split_enron["y_test"], "lr_enron")
    m_multi = compute_metrics(
        trained["logistic_regression"], split["X_test"], split["y_test"], "lr_multi"
    )
    interpret.comparative_table(m_enron, m_multi)

    log.info("=== 12) Auditoria adversarial (tokens de identidad Enron) ===")

    def _train_lr(X, y):
        m = build_models()["logistic_regression"]
        m.fit(X, y)
        return m

    interpret.adversarial_audit(
        _train_lr, split["X_train"], split["y_train"], split["X_test"], split["y_test"]
    )

    log.info("=== 13) Evaluacion cruzada leave-one-corpus-out (3 experimentos) ===")
    from .cross_eval import run_cross_eval
    run_cross_eval()

    path = save_model("best_model", best_model)
    with open(config.MODELS_DIR / "best_model.meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_name, "val_metrics": val_metrics, "test_metrics": test_metrics},
            f, indent=2, ensure_ascii=False,
        )
    log.info("Mejor modelo serializado en %s", path)

    elapsed = time.perf_counter() - t_start
    log.info("=== PIPELINE COMPLETO en %.1fs ===", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
