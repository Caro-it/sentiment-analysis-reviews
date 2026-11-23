"""Ruta de inferencia de producción — WeLoveReviews.

Lee data/raw/reviews.csv, predice el sentimiento de cada reseña con un modelo
preentrenado de Hugging Face y escribe data/processed/reviews_with_sentiment.csv.

Uso (desde la raíz del repo):
    python src/app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import pipeline

# Modelo fijado por nombre Y por commit: si alguien clona el repo dentro de un año,
# descargará exactamente los mismos pesos, no lo que "latest" sea en ese momento.
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

BATCH_SIZE = 32


def load_sentiment_model():
    """Carga el pipeline UNA sola vez. Los pesos se descargan a la caché de Hugging Face,
    no al repositorio."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
    )


def label_to_stars(label: str) -> int:
    """El modelo devuelve etiquetas tipo '4 stars' / '1 star' → 4 / 1."""
    return int(label.split()[0])


def stars_to_band(stars: int) -> str:
    """1–2 → negativo, 3 → neutral, 4–5 → positivo."""
    if stars <= 2:
        return "negativo"
    if stars == 3:
        return "neutral"
    return "positivo"


def rating_to_band(rating: int) -> str:
    """Misma regla aplicada a la puntuación que dio el cliente."""
    return stars_to_band(int(rating))


def predict(texts: list[str], classifier) -> pd.DataFrame:
    """Predice en lotes con el pipeline ya cargado (no lo recarga)."""
    results = classifier(texts, batch_size=BATCH_SIZE, truncation=True)
    stars = [label_to_stars(r["label"]) for r in results]
    return pd.DataFrame(
        {
            "pred_stars": stars,
            "pred_score": [round(float(r["score"]), 4) for r in results],
            "pred_band": [stars_to_band(s) for s in stars],
        }
    )


def enrich(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """Añade predicción, banda y marca de falso negativo a cada reseña."""
    preds = predict(df["review_text"].tolist(), classifier)
    out = pd.concat([df.reset_index(drop=True), preds], axis=1)
    out["rating_band"] = out["rating"].map(rating_to_band)
    out["model"] = f"{MODEL_NAME}@{MODEL_REVISION[:7]}"
    # Falso negativo: el cliente puso 4–5 estrellas pero el modelo predice 1–2.
    out["false_negative"] = (out["rating"] >= 4) & (out["pred_stars"] <= 2)
    return out


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    df["review_text"] = df["review_text"].astype(str).str.strip()

    classifier = load_sentiment_model()  # una sola vez, fuera de cualquier bucle
    enriched = enrich(df, classifier)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_PATH, index=False)

    dist = enriched["pred_band"].value_counts(normalize=True).mul(100).round(1)
    print(f"Reseñas procesadas: {len(enriched)}")
    print(f"Distribución predicha (%): {dist.to_dict()}")
    print(f"Falsos negativos (rating 4–5, modelo 1–2): {int(enriched['false_negative'].sum())}")
    print(f"Guardado en: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
