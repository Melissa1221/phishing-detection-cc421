from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from . import config
from .preprocess import clean_text

DEFAULT_MODEL = config.MODELS_DIR / "best_model.joblib"


def load_model(path=DEFAULT_MODEL):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {path}. "
            "Ejecuta primero `python -m src.run_pipeline`."
        )
    return joblib.load(path)


def predict_email(raw_text, model=None):
    if model is None:
        model = load_model()
    cleaned = clean_text(raw_text)
    proba = float(model.predict_proba([cleaned])[:, 1][0])
    label = int(proba >= 0.5)
    return {
        "label": label,
        "label_name": config.CLASS_NAMES[label],
        "phishing_proba": proba,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clasifica un correo como phishing o legitimo.")
    parser.add_argument("--text", type=str, help="Texto del correo a clasificar.")
    parser.add_argument("--file", type=str, help="Ruta a un archivo de texto con el correo.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)

    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()

    model = load_model(args.model)
    result = predict_email(raw, model=model)
    print(
        f"Prediccion: {result['label_name'].upper()} "
        f"(prob. phishing = {result['phishing_proba']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
