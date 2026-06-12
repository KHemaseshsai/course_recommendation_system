from __future__ import annotations

import math

import numpy as np
import pandas as pd


def precision_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    return sum(item in relevant_ids for item in top_k) / k


def recall_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    return sum(item in relevant_ids for item in top_k) / len(relevant_ids)


def average_precision(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended_ids[:k], start=1):
        if item in relevant_ids:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_ids), k)


def ndcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def catalog_coverage(recommendation_lists: list[list[str]], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    unique_recommended = {item for recommendations in recommendation_lists for item in recommendations}
    return min(len(unique_recommended) / catalog_size, 1.0)


def intra_list_diversity(recommended_ids: list[str], courses: pd.DataFrame) -> float:
    if len(recommended_ids) < 2 or courses.empty:
        return 0.0
    metadata = (
        courses.drop_duplicates(subset=["course"])
        .set_index("course")[["category", "sub_category"]]
        .to_dict("index")
    )
    distances = []
    for left_index, left_id in enumerate(recommended_ids):
        for right_id in recommended_ids[left_index + 1 :]:
            left = metadata.get(left_id)
            right = metadata.get(right_id)
            if not left or not right:
                continue
            similarity = 0.0
            if left["category"] == right["category"]:
                similarity += 0.6
            if left["sub_category"] == right["sub_category"]:
                similarity += 0.4
            distances.append(1.0 - similarity)
    return float(np.mean(distances)) if distances else 0.0


def popularity_bias(recommended_ids: list[str], courses: pd.DataFrame) -> float:
    if not recommended_ids or courses.empty or "popularity_score" not in courses:
        return 0.0
    popularity = (
        courses.groupby("course", as_index=True)["popularity_score"]
        .max()
    )
    values = [float(popularity.get(course, 0.0)) for course in recommended_ids]
    return float(np.mean(values)) if values else 0.0


def evaluate_ranking(
    recommended_ids: list[str],
    relevant_ids: set[str],
    courses: pd.DataFrame,
    k: int,
) -> dict[str, float]:
    return {
        "precision_at_k": precision_at_k(recommended_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(recommended_ids, relevant_ids, k),
        "map_at_k": average_precision(recommended_ids, relevant_ids, k),
        "ndcg_at_k": ndcg_at_k(recommended_ids, relevant_ids, k),
        "diversity": intra_list_diversity(recommended_ids[:k], courses),
        "popularity_bias": popularity_bias(recommended_ids[:k], courses),
    }
