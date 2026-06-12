from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recommender.interaction_store import InteractionStore


@dataclass
class CollaborativeFilteringResult:
    scores: np.ndarray
    backend: str
    user_count: int
    item_count: int
    interaction_count: int


class ImplicitMatrixFactorization:
    def __init__(self, store: InteractionStore | None = None, n_components: int = 24) -> None:
        self.store = store or InteractionStore()
        self.n_components = n_components

    def score(self, user_id: str, courses: pd.DataFrame) -> CollaborativeFilteringResult:
        interactions = self.store.training_frame()
        scores = np.zeros(len(courses), dtype="float32")
        if interactions.empty or not user_id:
            return CollaborativeFilteringResult(scores, "cold_start", 0, len(courses), 0)

        known_courses = courses["course"].astype(str).tolist()
        course_to_index = {course: index for index, course in enumerate(known_courses)}
        usable = interactions[interactions["course"].isin(course_to_index)].copy()
        if usable.empty or user_id not in set(usable["user_id"]):
            return CollaborativeFilteringResult(scores, "cold_start", usable["user_id"].nunique(), len(courses), len(usable))

        users = sorted(usable["user_id"].unique())
        user_to_index = {user: index for index, user in enumerate(users)}
        matrix = np.zeros((len(users), len(courses)), dtype="float32")
        for row in usable.itertuples(index=False):
            matrix[user_to_index[row.user_id], course_to_index[row.course]] = np.log1p(
                float(row.implicit_weight)
            )

        target_user_index = user_to_index[user_id]
        if len(users) < 2 or np.count_nonzero(matrix) < 4:
            scores = self._item_affinity_scores(matrix[target_user_index], courses)
            return CollaborativeFilteringResult(scores, "implicit_item_affinity", len(users), len(courses), len(usable))

        component_count = min(self.n_components, min(matrix.shape) - 1)
        if component_count < 1:
            scores = self._item_affinity_scores(matrix[target_user_index], courses)
            return CollaborativeFilteringResult(scores, "implicit_item_affinity", len(users), len(courses), len(usable))

        model = TruncatedSVD(n_components=component_count, random_state=42)
        user_factors = model.fit_transform(matrix)
        item_factors = model.components_.T
        scores = np.matmul(user_factors[target_user_index], item_factors.T).astype("float32")
        consumed = matrix[target_user_index] > 0
        scores[consumed] *= 0.25
        return CollaborativeFilteringResult(_normalize(scores), "truncated_svd", len(users), len(courses), len(usable))

    def _item_affinity_scores(self, user_vector: np.ndarray, courses: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(courses), dtype="float32")
        interacted_indices = np.flatnonzero(user_vector > 0)
        if len(interacted_indices) == 0:
            return scores

        interacted = courses.iloc[interacted_indices]
        categories = set(interacted["category"].dropna().astype(str).str.lower())
        subcategories = set(interacted["sub_category"].dropna().astype(str).str.lower())
        platforms = set(interacted["site"].dropna().astype(str).str.lower())
        for index, row in enumerate(courses.itertuples(index=False)):
            score = 0.0
            if str(row.category).lower() in categories:
                score += 0.45
            if str(row.sub_category).lower() in subcategories:
                score += 0.35
            if str(row.site).lower() in platforms:
                score += 0.20
            scores[index] = score
        scores[interacted_indices] *= 0.25
        return np.clip(scores, 0.0, 1.0)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        return np.zeros_like(values, dtype="float32")
    return ((values - minimum) / (maximum - minimum)).astype("float32")


def score_collaborative(user_id: str, courses: pd.DataFrame) -> CollaborativeFilteringResult:
    return ImplicitMatrixFactorization().score(user_id, courses)
