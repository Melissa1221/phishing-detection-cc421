from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from . import config
from .preprocess import clean_text

DEFAULT_MODEL = config.MODELS_DIR / "best_model.joblib"
DEFAULT_THRESHOLD = 0.5


def load_model(path=DEFAULT_MODEL):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {path}. "
            "Ejecuta primero `python -m src.run_pipeline`."
        )
    return joblib.load(path)


def predict_email(raw_text, model=None, threshold=DEFAULT_THRESHOLD):
    if model is None:
        model = load_model()
    cleaned = clean_text(raw_text)
    proba = float(model.predict_proba([cleaned])[:, 1][0])
    label = int(proba >= threshold)
    return {
        "label": label,
        "label_name": config.CLASS_NAMES[label],
        "phishing_proba": proba,
        "threshold": threshold,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clasifica un correo como phishing o legitimo.")
    parser.add_argument("--text", type=str, help="Texto del correo a clasificar.")
    parser.add_argument("--file", type=str, help="Ruta a un archivo de texto con el correo.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Umbral de decision (default 0.5). Bajar para mas sensibilidad.",
    )
    args = parser.parse_args(argv)

    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()

    model = load_model(args.model)
    result = predict_email(raw, model=model, threshold=args.threshold)
    print(
        f"Prediccion: {result['label_name'].upper()} "
        f"(prob. phishing = {result['phishing_proba']:.4f}, "
        f"threshold = {result['threshold']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
