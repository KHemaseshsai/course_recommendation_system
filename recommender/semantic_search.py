from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
EMBEDDINGS_PATH = ARTIFACT_DIR / "course_embeddings.npy"
FAISS_INDEX_PATH = ARTIFACT_DIR / "faiss.index"


@dataclass
class SemanticSearchResult:
    scores: np.ndarray | None
    backend: str


class SemanticCourseIndex:
    def __init__(self, courses: pd.DataFrame) -> None:
        self.courses = courses
        self.backend = "tfidf"
        self.model = None
        self.index = None
        self.embeddings: np.ndarray | None = None
        self._build()

    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return

        self.model = SentenceTransformer(MODEL_NAME)
        ARTIFACT_DIR.mkdir(exist_ok=True)
        if EMBEDDINGS_PATH.exists():
            embeddings = np.load(EMBEDDINGS_PATH)
            if len(embeddings) == len(self.courses):
                self.embeddings = embeddings.astype("float32")
            else:
                self.embeddings = None

        if self.embeddings is None:
            texts = self.courses["content"].fillna("").astype(str).tolist()
            self.embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")
            np.save(EMBEDDINGS_PATH, self.embeddings)

        try:
            import faiss
        except ImportError:
            self.backend = "sentence-transformers"
            return

        dimension = int(self.embeddings.shape[1])
        index = None
        if FAISS_INDEX_PATH.exists():
            try:
                stored_index = faiss.read_index(str(FAISS_INDEX_PATH))
                if stored_index.d == dimension and stored_index.ntotal == len(self.embeddings):
                    index = stored_index
            except RuntimeError:
                index = None
        if index is None:
            index = faiss.IndexFlatIP(dimension)
            index.add(self.embeddings)
            faiss.write_index(index, str(FAISS_INDEX_PATH))
        self.index = index
        self.backend = "faiss"

    def score(self, query: str, candidate_count: int = 250) -> SemanticSearchResult:
        if not query or self.model is None or self.embeddings is None:
            return SemanticSearchResult(scores=None, backend=self.backend)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores = np.zeros(len(self.courses), dtype="float32")
        if self.index is not None:
            top_k = min(candidate_count, len(self.courses))
            similarities, indices = self.index.search(query_embedding, top_k)
            valid_indices = indices[0][indices[0] >= 0]
            scores[valid_indices] = similarities[0][: len(valid_indices)]
        else:
            scores = np.matmul(self.embeddings, query_embedding[0])

        return SemanticSearchResult(scores=np.clip(scores, 0.0, 1.0), backend=self.backend)


@lru_cache(maxsize=1)
def get_semantic_index(course_count: int, first_course: str, last_course: str) -> SemanticCourseIndex:
    from api import _load_assets

    courses, _, _, _ = _load_assets()
    if len(courses) != course_count or str(courses.iloc[0]["course"]) != first_course:
        courses = courses.copy()
    return SemanticCourseIndex(courses)


def score_courses(courses: pd.DataFrame, query: str) -> SemanticSearchResult:
    if courses.empty:
        return SemanticSearchResult(scores=None, backend="empty")
    index = get_semantic_index(
        len(courses),
        str(courses.iloc[0]["course"]),
        str(courses.iloc[-1]["course"]),
    )
    return index.score(query)
