from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from recommender.ranking import get_hybrid_weights
from recommender.semantic_search import EMBEDDINGS_PATH, FAISS_INDEX_PATH, MODEL_NAME


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Online_Courses.csv"
RECOMMENDATION_VERSION = "4.0.0"
RANKING_VERSION = "behavioral-hybrid-v2"


def _fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def model_metadata() -> dict[str, object]:
    weights = get_hybrid_weights()
    ranking_payload = json.dumps(weights, sort_keys=True).encode("utf-8")
    return {
        "recommendation_version": RECOMMENDATION_VERSION,
        "ranking_version": RANKING_VERSION,
        "sentence_transformer_model": MODEL_NAME,
        "dataset_version": _fingerprint(DATASET_PATH),
        "embeddings_version": _fingerprint(EMBEDDINGS_PATH),
        "faiss_index_version": _fingerprint(FAISS_INDEX_PATH),
        "ranking_config_version": hashlib.sha256(ranking_payload).hexdigest()[:16],
        "ranking_weights": weights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
