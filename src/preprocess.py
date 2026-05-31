from __future__ import annotations

import html
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords

from . import config

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
_NUM = re.compile(r"\b\d[\d.,]*\b")
_NON_ALPHA = re.compile(r"[^a-z_\s]")
_MULTISPACE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _get_stopwords():
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))
    sw.discard(config.URL_TOKEN)
    sw.discard(config.EMAIL_TOKEN)
    sw.discard(config.NUM_TOKEN)
    return frozenset(sw)


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str) or not text:
        return ""

    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    # url va antes que email porque las urls pueden tener @, y email antes que num
    text = _URL.sub(f" {config.URL_TOKEN} ", text)
    text = _EMAIL.sub(f" {config.EMAIL_TOKEN} ", text)
    text = _NUM.sub(f" {config.NUM_TOKEN} ", text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()

    if remove_stopwords:
        sw = _get_stopwords()
        text = " ".join(tok for tok in text.split() if tok not in sw and len(tok) > 1)

    return text


def preprocess_series(series, remove_stopwords: bool = True):
    return series.map(lambda t: clean_text(t, remove_stopwords=remove_stopwords))


if __name__ == "__main__":
    sample = (
        "URGENT! Your <b>account</b> will be suspended. "
        "Verify at http://phish.example.com/login or email admin@bank.com. "
        "Call 1-800-555-0199 now!!!"
    )
    print("IN :", sample)
    print("OUT:", clean_text(sample))
