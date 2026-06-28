from __future__ import annotations

import argparse
import json
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
)
from .features import stratified_split
from .preprocess import preprocess_series
from .train import build_models, save_model, train_model


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

    set_seeds()
    t_start = time.perf_counter()

    print("\n=== 1) Carga y consolidacion del corpus ===")
    df = load_corpus()
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=config.RANDOM_STATE)
        df = df.reset_index(drop=True)
        print(f"[pipeline] Muestreo de depuracion: {len(df)} correos.")

    print("\n=== 2) Limpieza y normalizacion ===")
    t0 = time.perf_counter()
    df["clean_text"] = preprocess_series(df["text"])
    # al limpiar algunos quedan vacios (solo stopwords o simbolos)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"[pipeline] Limpieza completada en {time.perf_counter() - t0:.1f}s "
          f"({len(df)} correos no vacios).")
    df.to_csv(config.PROCESSED_DIR / "corpus_clean.csv", index=False)

    print("\n=== 3) Analisis exploratorio (EDA) ===")
    eda_summary = eda.run_eda(df, clean_col="clean_text")
    print(json.dumps(eda_summary["class_distribution"], indent=2, ensure_ascii=False))
    print(json.dumps(eda_summary["length_stats"], indent=2, ensure_ascii=False))

    print("\n=== 4) Particion estratificada 70/15/15 ===")
    split = stratified_split(
        df["clean_text"].values, df["label"].values, source=df["source"].values
    )
    print(
        f"[pipeline] train={len(split['X_train'])}, "
        f"val={len(split['X_val'])}, test={len(split['X_test'])}"
    )

    print("\n=== 5) Entrenamiento de modelos ===")
    models = build_models()
    trained = {}
    train_times = {}
    cv_results = {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, model in models.items():
        fitted, elapsed = train_model(name, model, split["X_train"], split["y_train"])
        trained[name] = fitted
        train_times[name] = elapsed
        if not args.no_cv:
            scores = cross_val_score(
                build_models()[name],  # pipeline nuevo para que el CV sea honesto
                split["X_train"],
                split["y_train"],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )
            cv_results[name] = {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}
            print(f"[cv] {name}: F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

    print("\n=== 6) Evaluacion en validacion ===")
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
        print(
            f"[val] {name}: F1={m['f1']:.4f} recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} PR-AUC={m['pr_auc']:.4f}"
        )

    best_name = max(val_metrics, key=lambda d: d["f1"])["model"]
    best_model = trained[best_name]
    print(f"\n[pipeline] Mejor modelo por F1 en validacion: {best_name}")

    save_roc_pr_curves(val_scores_for_curves, split["y_val"])
    for name, model in trained.items():
        save_confusion_matrix(model, split["X_val"], split["y_val"], name)
    save_metrics_json(val_metrics, "val_metrics.json")

    # evaluacion final en test, una sola vez
    print("\n=== 7) Evaluacion final en TEST (mejor modelo) ===")
    test_metrics = compute_metrics(best_model, split["X_test"], split["y_test"], best_name)
    targets = meets_targets(test_metrics)
    test_metrics["targets_met"] = targets
    save_confusion_matrix(best_model, split["X_test"], split["y_test"], f"{best_name}_TEST")
    save_metrics_json([test_metrics], "test_metrics.json")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
                     indent=2, ensure_ascii=False))

    print("\n=== 8) Curva de aprendizaje (modelo ganador) ===")
    save_learning_curve(best_name, build_models()[best_name], split["X_train"], split["y_train"])

    print("\n=== 9) Interpretabilidad por pesos del modelo ===")
    if "logistic_regression" in trained:
        interpret.save_top_weighted_terms(trained["logistic_regression"])
    else:
        print("[pipeline] logistic_regression no entrenado; se omite analisis de pesos.")

    print("\n=== 10) Analisis de errores ===")
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

    print("\n=== 11) Tabla comparativa Enron-solo vs multi-fuente ===")
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

    print("\n=== 12) Auditoria adversarial (tokens de identidad Enron) ===")

    def _train_lr(X, y):
        m = build_models()["logistic_regression"]
        m.fit(X, y)
        return m

    interpret.adversarial_audit(
        _train_lr, split["X_train"], split["y_train"], split["X_test"], split["y_test"]
    )

    path = save_model("best_model", best_model)
    with open(config.MODELS_DIR / "best_model.meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_name, "val_metrics": val_metrics, "test_metrics": test_metrics},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[pipeline] Mejor modelo serializado en {path}")

    print(f"\n=== PIPELINE COMPLETO en {time.perf_counter() - t_start:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
