from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def class_distribution(df):
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "total": total,
        "legit": int(counts.get(config.LABEL_LEGIT, 0)),
        "phishing": int(counts.get(config.LABEL_PHISHING, 0)),
        "phishing_ratio": float(counts.get(config.LABEL_PHISHING, 0) / total),
    }


def length_stats(df, text_col="text"):
    lengths = df[text_col].str.split().map(len)
    return {
        "mean_tokens": float(lengths.mean()),
        "median_tokens": float(lengths.median()),
        "p95_tokens": float(lengths.quantile(0.95)),
        "max_tokens": int(lengths.max()),
    }


def save_class_distribution_fig(df):
    dist = class_distribution(df)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(
        config.CLASS_NAMES,
        [dist["legit"], dist["phishing"]],
        color=["#2a9d8f", "#e76f51"],
    )
    ax.set_ylabel("Nro de correos")
    ax.set_title("Distribucion de clases")
    for i, v in enumerate([dist["legit"], dist["phishing"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = config.FIGURES_DIR / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_length_distribution_fig(df, text_col="text"):
    lengths = df[text_col].str.split().map(len)
    cap = int(lengths.quantile(0.99))  # recortamos la cola larga para que se vea
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, color, name in (
        (config.LABEL_LEGIT, "#2a9d8f", "legitimo"),
        (config.LABEL_PHISHING, "#e76f51", "phishing"),
    ):
        subset = lengths[df["label"] == label].clip(upper=cap)
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=name)
    ax.set_xlabel("Longitud del correo (tokens)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de longitudes por clase")
    ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / "length_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def top_words_by_class(df, clean_col, top_n=20):
    result = {}
    for label, name in ((config.LABEL_LEGIT, "legit"), (config.LABEL_PHISHING, "phishing")):
        counter = Counter()
        for txt in df.loc[df["label"] == label, clean_col]:
            counter.update(txt.split())
        result[name] = counter.most_common(top_n)
    return result


def save_top_words_fig(top_words, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, color) in zip(axes, (("legit", "#2a9d8f"), ("phishing", "#e76f51"))):
        items = top_words[name][:top_n][::-1]
        words = [w for w, _ in items]
        freqs = [c for _, c in items]
        ax.barh(words, freqs, color=color)
        ax.set_title(f"Top {top_n} palabras - {name}")
        ax.set_xlabel("Frecuencia")
    fig.tight_layout()
    path = config.FIGURES_DIR / "top_words.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def run_eda(df, clean_col="clean_text"):
    summary = {
        "class_distribution": class_distribution(df),
        "length_stats": length_stats(df),
        "figures": {
            "class_distribution": save_class_distribution_fig(df),
            "length_distribution": save_length_distribution_fig(df),
        },
    }
    if clean_col in df.columns:
        tw = top_words_by_class(df, clean_col)
        summary["top_words"] = tw
        summary["figures"]["top_words"] = save_top_words_fig(tw)

    with open(config.METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
