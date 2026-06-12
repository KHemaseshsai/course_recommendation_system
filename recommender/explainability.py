from __future__ import annotations

import pandas as pd


def explanation_bullets(row: pd.Series, profile: dict[str, str], semantic_backend: str) -> list[str]:
    bullets = []
    if row.get("semantic_score", 0.0) >= 0.35:
        bullets.append(f"Matches the goal using {semantic_backend} semantic retrieval")
    elif row.get("goal_fit", 0.0) >= 0.2:
        bullets.append("Matches important keywords and goal profile signals")

    if row.get("level_fit", 0.0) >= 0.95:
        bullets.append(f"Fits {profile['level']} skill level")
    if row.get("budget_fit", 0.0) >= 0.95 and profile["budget_preference"] != "Any":
        bullets.append(f"Fits {profile['budget_preference'].lower()} budget preference")
    if row.get("delivery_fit", 0.0) >= 0.95 and profile["delivery_preference"] != "Any":
        bullets.append(f"Matches {profile['delivery_preference'].lower()} delivery preference")
    if row.get("popularity_score", 0.0) >= 0.6:
        bullets.append("Has strong learner popularity signal")
    if row.get("quality_score", 0.0) >= 0.85:
        bullets.append("Has high rating signal")
    if row.get("behavioral_profile_score", 0.0) >= 0.25:
        bullets.append("Matches categories and progression inferred from your learning history")
    if row.get("collaborative_score", 0.0) >= 0.35:
        bullets.append("Learners with similar implicit feedback favored this course")

    return bullets[:4] or ["Solid fit across goal, preference, popularity, and rating signals"]
