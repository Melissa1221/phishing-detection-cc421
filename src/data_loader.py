from __future__ import annotations

import email as email_mod
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from . import config

ENRON_URL = (
    "https://raw.githubusercontent.com/MWiechmann/"
    "enron_spam_data/master/enron_spam_data.zip"
)
ENRON_ZIP = config.RAW_DIR / "enron_spam_data.zip"
ENRON_CSV = config.RAW_DIR / "enron_spam_data.csv"

# SpamAssassin
SA_BASE = "https://spamassassin.apache.org/old/publiccorpus"
SA_ARCHIVES = [
    ("20021010_easy_ham.tar.bz2", "easy_ham", config.LABEL_LEGIT),
    ("20021010_hard_ham.tar.bz2", "hard_ham", config.LABEL_LEGIT),
    ("20030228_spam.tar.bz2", "spam", config.LABEL_PHISHING),
    ("20050311_spam_2.tar.bz2", "spam_2", config.LABEL_PHISHING),
]
SA_DIR = config.RAW_DIR / "spamassassin"

# Nazario
NAZARIO_URL = (
    "https://github.com/diegoocampoh/MachineLearningPhishing"
    "/raw/master/code/resources/emails-phishing.mbox"
)
NAZARIO_MBOX = config.RAW_DIR / "nazario_phishing.mbox"

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


# SpamAssain
def _extract_msg_content(msg) -> tuple[str, str]:
    try:
        subject = str(msg.get("Subject", "") or "")
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode("utf-8", errors="replace"))
        return subject, " ".join(parts)
    except Exception:
        return "", ""


def _parse_rfc822(path: Path) -> tuple[str, str]:
    try:
        msg = email_mod.message_from_bytes(path.read_bytes())
        return _extract_msg_content(msg)
    except Exception:
        return "", ""


def _download_spamassassin():
    SA_DIR.mkdir(exist_ok=True)
    for fname, folder, _ in SA_ARCHIVES:
        dest = SA_DIR / folder
        if dest.exists():
            continue
        archive = config.RAW_DIR / fname
        if not archive.exists():
            url = f"{SA_BASE}/{fname}"
            print(f"[data_loader] Descargando {fname} ...")
            with urlopen(url, timeout=180) as resp:
                archive.write_bytes(resp.read())
        print(f"[data_loader] Extrayendo {fname} ...")
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(SA_DIR)


def load_spamassassin():
    _download_spamassassin()
    records = []
    for _, folder, label in SA_ARCHIVES:
        folder_path = SA_DIR / folder
        if not folder_path.exists():
            print(f"[data_loader] Advertencia: carpeta no encontrada {folder_path}")
            continue
        for fpath in sorted(folder_path.iterdir()):
            # cmds son instrucciones
            if not fpath.is_file() or fpath.name == "cmds":
                continue
            subject, message = _parse_rfc822(fpath)
            records.append({"subject": subject, "message": message, "label": label})
    df = pd.DataFrame(records)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["message"] = df["message"].fillna("").astype(str)
    df["source"] = "spamassassin"
    print(f"[data_loader] SpamAssassin: {len(df)} correos cargados.")
    return df[CANONICAL_COLS + ["source"]].reset_index(drop=True)

# Nazario

def _download_nazario():
    if NAZARIO_MBOX.exists():
        return
    print("[data_loader] Descargando corpus Nazario (mirror GitHub) ...")
    with urlopen(NAZARIO_URL, timeout=180) as resp:
        NAZARIO_MBOX.write_bytes(resp.read())


def _iter_mbox_bytes(data: bytes):

    current: list[bytes] = []
    for line in data.splitlines(keepends=True):
        if line.startswith(b"From ") and current:
            yield email_mod.message_from_bytes(b"".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        yield email_mod.message_from_bytes(b"".join(current))


def load_nazario():
    _download_nazario()
    data = NAZARIO_MBOX.read_bytes()
    records = []
    for msg in _iter_mbox_bytes(data):
        subject, message = _extract_msg_content(msg)
        records.append({
            "subject": subject,
            "message": message,
            "label": config.LABEL_PHISHING,
        })
    df = pd.DataFrame(records)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["message"] = df["message"].fillna("").astype(str)
    df["source"] = "nazario"
    print(f"[data_loader] Nazario: {len(df)} correos de phishing cargados.")
    return df[CANONICAL_COLS + ["source"]].reset_index(drop=True)


# Corpus
def load_corpus(deduplicate: bool = True):
    frames = [load_enron(), load_spamassassin(), load_nazario()]
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
    print()
    print("Conteo por fuente y clase (0=legitimo, 1=phishing):")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print()
    print("Tasa de phishing por fuente:")
    print(df.groupby("source")["label"].mean().rename("tasa_phishing").round(3))
