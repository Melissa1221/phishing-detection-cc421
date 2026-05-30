from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 1 = phishing/spam, 0 = legitimo (ham)
LABEL_PHISHING = 1
LABEL_LEGIT = 0
CLASS_NAMES = ["legitimo", "phishing"]

TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5

TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 2)

URL_TOKEN = "url_token"
EMAIL_TOKEN = "email_token"
NUM_TOKEN = "num_token"

TARGET_RECALL = 0.90
TARGET_F1 = 0.88
TARGET_PR_AUC = 0.92
