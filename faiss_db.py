import json
import os
import re

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "faiss-cpu is required to build the FAISS index. Install it with: pip install faiss-cpu"
    ) from e


EMBED_DIM = 64
STORE_DIR = "weather_store"
INDEX_PATH = os.path.join(STORE_DIR, "weather.index")
DOCS_PATH = os.path.join(STORE_DIR, "weather_docs.json")

WEATHER_DOCS = [
    {
        "city": "Karachi",
        "temperature_c": 31,
        "condition": "Humid",
        "humidity": 68,
        "text": "Karachi weather is humid with temperature 31C and humidity 68 percent."
    },
    {
        "city": "Lahore",
        "temperature_c": 34,
        "condition": "Sunny",
        "humidity": 40,
        "text": "Lahore weather is sunny with temperature 34C and humidity 40 percent."
    },
    {
        "city": "Islamabad",
        "temperature_c": 27,
        "condition": "Cloudy",
        "humidity": 52,
        "text": "Islamabad weather is cloudy with temperature 27C and humidity 52 percent."
    },
    {
        "city": "London",
        "temperature_c": 16,
        "condition": "Rainy",
        "humidity": 74,
        "text": "London weather is rainy with temperature 16C and humidity 74 percent."
    },
    {
        "city": "New York",
        "temperature_c": 21,
        "condition": "Partly Cloudy",
        "humidity": 58,
        "text": "New York weather is partly cloudy with temperature 21C and humidity 58 percent."
    },
]


def hash_token(token: str, dim: int = EMBED_DIM) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(token)) % dim


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        return vec
    for tok in tokens:
        vec[hash_token(tok, dim)] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def main() -> None:
    os.makedirs(STORE_DIR, exist_ok=True)

    vectors = np.vstack([embed_text(doc["text"]) for doc in WEATHER_DOCS]).astype(np.float32)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(WEATHER_DOCS, f, ensure_ascii=False, indent=2)

    print(f"Built FAISS weather DB")
    print(f"Index: {INDEX_PATH}")
    print(f"Docs: {DOCS_PATH}")
    print(f"Vectors: {len(WEATHER_DOCS)}")


if __name__ == "__main__":
    main()