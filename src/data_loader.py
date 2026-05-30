from __future__ import annotations

import zipfile
from urllib.request import urlopen

import pandas as pd

from . import config

ENRON_URL = (
    "https://raw.githubusercontent.com/MWiechmann/"
    "enron_spam_data/master/enron_spam_data.zip"
)
ENRON_ZIP = config.RAW_DIR / "enron_spam_data.zip"
ENRON_CSV = config.RAW_DIR / "enron_spam_data.csv"

CANONICAL_COLS = ["subject", "message", "label"]


def _download_enron():
    if ENRON_CSV.exists():
        return
    if not ENRON_ZIP.exists():
        print(f"[data_loader] Descargando Enron-Spam desde {ENRON_URL} ...")
        with urlopen(ENRON_URL, timeout=120) as resp:
            ENRON_ZIP.write_bytes(resp.read())
    print("[data_loader] Extrayendo Enron-Spam ...")
    with zipfile.ZipFile(ENRON_ZIP) as zf:
        zf.extractall(config.RAW_DIR)


def load_enron():
    _download_enron()
    df = pd.read_csv(ENRON_CSV)
    df = df.rename(
        columns={
            "Subject": "subject",
            "Message": "message",
            "Spam/Ham": "label_str",
        }
    )
    df["subject"] = df["subject"].fillna("").astype(str)
    df["message"] = df["message"].fillna("").astype(str)
    label_map = {"spam": config.LABEL_PHISHING, "ham": config.LABEL_LEGIT}
    df["label"] = df["label_str"].str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["source"] = "enron"
    return df[CANONICAL_COLS + ["source"]].reset_index(drop=True)


def load_corpus(deduplicate: bool = True):
    frames = [load_enron()]
    corpus = pd.concat(frames, ignore_index=True)

    corpus["text"] = (corpus["subject"] + " " + corpus["message"]).str.strip()

    before = len(corpus)
    corpus = corpus[corpus["text"].str.len() > 0].reset_index(drop=True)
    n_empty = before - len(corpus)

    n_dups = 0
    if deduplicate:
        before = len(corpus)
        corpus = corpus.drop_duplicates(subset=["text"]).reset_index(drop=True)
        n_dups = before - len(corpus)

    print(
        f"[data_loader] Corpus consolidado: {len(corpus)} correos "
        f"({n_empty} vacios y {n_dups} duplicados removidos)."
    )
    return corpus


if __name__ == "__main__":
    df = load_corpus()
    print(df["label"].value_counts(normalize=True).rename("proporcion"))
    print(df.head())
