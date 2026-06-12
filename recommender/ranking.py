from __future__ import annotations

import json
import os

import pandas as pd


DEFAULT_HYBRID_WEIGHTS = {
    "semantic_similarity": 0.32,
    "user_preference_match": 0.22,
    "behavioral_profile_score": 0.16,
    "popularity_score": 0.10,
    "rating_score": 0.08,
    "collaborative_score": 0.12,
}


def get_hybrid_weights() -> dict[str, float]:
    configured = os.getenv("RECOMMENDER_WEIGHTS", "").strip()
    weights = DEFAULT_HYBRID_WEIGHTS.copy()
    if configured:
        parsed = json.loads(configured)
        weights.update({key: float(value) for key, value in parsed.items() if key in weights})
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        raise ValueError("At least one recommender weight must be positive")
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def compute_user_preference_match(courses: pd.DataFrame) -> pd.Series:
    preference_columns = [
        "level_fit",
        "time_fit",
        "completion_fit",
        "style_fit",
        "budget_fit",
        "delivery_fit",
        "certificate_fit",
        "platform_fit",
        "language_fit",
        "category_fit",
    ]
    available_columns = [column for column in preference_columns if column in courses.columns]
    if not available_columns:
        return pd.Series([0.0] * len(courses), index=courses.index)
    return courses[available_columns].mean(axis=1).clip(0.0, 1.0)


def compute_hybrid_score(
    courses: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    active_weights = weights or get_hybrid_weights()
    semantic = courses["semantic_score"].clip(0.0, 1.0)
    preferences = courses["user_preference_match"].clip(0.0, 1.0)
    behavioral = courses.get(
        "behavioral_profile_score",
        pd.Series([0.0] * len(courses), index=courses.index),
    ).clip(0.0, 1.0)
    popularity = courses["popularity_score"].clip(0.0, 1.0)
    rating = courses["quality_score"].clip(0.0, 1.0)
    collaborative = courses.get("collaborative_score", pd.Series([0.0] * len(courses), index=courses.index)).clip(0.0, 1.0)
    return (
        semantic * active_weights["semantic_similarity"]
        + preferences * active_weights["user_preference_match"]
        + behavioral * active_weights["behavioral_profile_score"]
        + popularity * active_weights["popularity_score"]
        + rating * active_weights["rating_score"]
        + collaborative * active_weights["collaborative_score"]
    )
