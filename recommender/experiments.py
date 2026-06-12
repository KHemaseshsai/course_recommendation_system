from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from recommender.evaluation import catalog_coverage, evaluate_ranking
from recommender.model_registry import model_metadata
from recommender.ranking import compute_hybrid_score
from recommender.semantic_search import score_courses


MODEL_NAMES = [
    "tfidf",
    "semantic_search",
    "hybrid",
    "collaborative_filtering",
    "hybrid_collaborative",
]


def _benchmark_scenarios(courses: pd.DataFrame, count: int = 8) -> list[dict[str, object]]:
    eligible = courses.groupby(["category", "sub_category"]).size().sort_values(ascending=False)
    scenarios = []
    for category, sub_category in eligible.index:
        relevant = courses[
            courses["category"].eq(category) & courses["sub_category"].eq(sub_category)
        ]["course"].tolist()
        if len(relevant) < 10:
            continue
        scenarios.append(
            {
                "query": f"{sub_category} {category}",
                "category": category,
                "sub_category": sub_category,
                "relevant": set(relevant),
            }
        )
        if len(scenarios) >= count:
            break
    return scenarios


def _synthetic_collaborative_scores(
    courses: pd.DataFrame,
    target_sub_category: str,
    random_seed: int,
) -> np.ndarray:
    sub_categories = courses["sub_category"].fillna("").astype(str)
    common = sub_categories.value_counts().head(20).index.tolist()
    if target_sub_category not in common:
        common.append(target_sub_category)
    matrix = np.zeros((len(common), len(courses)), dtype="float32")
    random = np.random.default_rng(random_seed)
    for user_index, sub_category in enumerate(common):
        candidates = np.flatnonzero(sub_categories.eq(sub_category).to_numpy())
        if not len(candidates):
            continue
        selected = random.choice(candidates, size=min(12, len(candidates)), replace=False)
        matrix[user_index, selected] = random.uniform(1.0, 5.0, size=len(selected))
    target_index = common.index(target_sub_category)
    components = min(12, min(matrix.shape) - 1)
    if components < 1 or np.count_nonzero(matrix) < 4:
        return np.zeros(len(courses), dtype="float32")
    model = TruncatedSVD(n_components=components, random_state=random_seed)
    user_factors = model.fit_transform(matrix)
    scores = user_factors[target_index] @ model.components_
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum <= minimum:
        return np.zeros(len(courses), dtype="float32")
    return ((scores - minimum) / (maximum - minimum)).astype("float32")


def run_benchmark(
    courses: pd.DataFrame,
    vectorizer: object,
    tfidf_matrix: object,
    k: int = 10,
    random_seed: int = 42,
) -> dict[str, object]:
    scenarios = _benchmark_scenarios(courses)
    model_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in MODEL_NAMES}
    recommendation_lists: dict[str, list[list[str]]] = {name: [] for name in MODEL_NAMES}

    for scenario_index, scenario in enumerate(scenarios):
        query = str(scenario["query"])
        tfidf_scores = cosine_similarity(vectorizer.transform([query]), tfidf_matrix).flatten()
        semantic_result = score_courses(courses, query)
        semantic_scores = semantic_result.scores if semantic_result.scores is not None else tfidf_scores
        collaborative_scores = _synthetic_collaborative_scores(
            courses,
            str(scenario["sub_category"]),
            random_seed + scenario_index,
        )

        features = courses.copy()
        features["semantic_score"] = semantic_scores
        features["user_preference_match"] = (
            features["category"].eq(scenario["category"]).astype(float) * 0.4
            + features["sub_category"].eq(scenario["sub_category"]).astype(float) * 0.6
        )
        features["behavioral_profile_score"] = features["user_preference_match"]
        features["quality_score"] = features["rating_value"].clip(0.0, 5.0) / 5.0
        features["collaborative_score"] = collaborative_scores
        hybrid_scores = compute_hybrid_score(
            features,
            {
                "semantic_similarity": 0.44,
                "user_preference_match": 0.28,
                "behavioral_profile_score": 0.0,
                "popularity_score": 0.18,
                "rating_score": 0.10,
                "collaborative_score": 0.0,
            },
        )
        hybrid_collaborative_scores = compute_hybrid_score(features)

        score_sets = {
            "tfidf": tfidf_scores,
            "semantic_search": semantic_scores,
            "hybrid": hybrid_scores.to_numpy(),
            "collaborative_filtering": collaborative_scores,
            "hybrid_collaborative": hybrid_collaborative_scores.to_numpy(),
        }
        for model_name, scores in score_sets.items():
            order = np.argsort(np.asarray(scores))[::-1][:k]
            recommendations = courses.iloc[order]["course"].tolist()
            recommendation_lists[model_name].append(recommendations)
            model_metrics[model_name].append(
                evaluate_ranking(recommendations, scenario["relevant"], courses, k)
            )

    results = []
    for model_name in MODEL_NAMES:
        metrics = model_metrics[model_name]
        averaged = {
            metric: round(float(np.mean([row[metric] for row in metrics])), 4)
            for metric in metrics[0]
        } if metrics else {}
        averaged["coverage"] = round(
            catalog_coverage(recommendation_lists[model_name], len(courses)),
            4,
        )
        results.append({"model": model_name, **averaged})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "deterministic catalog relevance with synthetic implicit-feedback personas",
        "random_seed": random_seed,
        "k": k,
        "scenario_count": len(scenarios),
        "catalog_size": len(courses),
        "models": results,
        "versions": model_metadata(),
    }


def write_reports(
    report: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    html_path = output_dir / "evaluation_dashboard.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(_dashboard_html(report), encoding="utf-8")
    return json_path, html_path


def _dashboard_html(report: dict[str, object]) -> str:
    models = report["models"]
    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k",
        "coverage",
        "diversity",
        "popularity_bias",
    ]
    cards = []
    for metric in metric_names:
        rows = []
        maximum = max((float(model.get(metric, 0.0)) for model in models), default=1.0) or 1.0
        for model in models:
            value = float(model.get(metric, 0.0))
            width = max(2.0, value / maximum * 100)
            rows.append(
                f"<div class='row'><span>{html.escape(str(model['model']))}</span>"
                f"<div class='track'><div class='bar' style='width:{width:.1f}%'></div></div>"
                f"<strong>{value:.4f}</strong></div>"
            )
        cards.append(f"<section><h2>{html.escape(metric)}</h2>{''.join(rows)}</section>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Recommender Evaluation Dashboard</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f6fb;color:#18212f;margin:0;padding:32px}}
main{{max-width:1200px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
section{{background:white;border-radius:14px;padding:20px;box-shadow:0 8px 30px #1f293714}}
.row{{display:grid;grid-template-columns:160px 1fr 62px;gap:10px;align-items:center;margin:12px 0;font-size:13px}}
.track{{height:18px;background:#e8ecf4;border-radius:9px;overflow:hidden}} .bar{{height:100%;background:#3867d6}}
h1{{margin-bottom:6px}} p{{color:#5c6678}} strong{{text-align:right}}
</style></head><body><main>
<h1>Recommender Evaluation Dashboard</h1>
<p>Generated {html.escape(str(report['generated_at']))}. Protocol: {html.escape(str(report['protocol']))}.
K={report['k']}, scenarios={report['scenario_count']}, catalog={report['catalog_size']}.</p>
<div class="grid">{''.join(cards)}</div>
</main></body></html>"""
