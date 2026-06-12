from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from recommender.interaction_store import InteractionStore


@dataclass
class UserProfile:
    user_id: str
    viewed_courses: list[str] = field(default_factory=list)
    clicked_courses: list[str] = field(default_factory=list)
    saved_courses: list[str] = field(default_factory=list)
    enrolled_courses: list[str] = field(default_factory=list)
    completed_courses: list[str] = field(default_factory=list)
    preferred_categories: list[str] = field(default_factory=list)
    budget_preference: str = "Any"
    skill_level: str = "Beginner"


class UserProfileStore:
    def __init__(self, interactions: InteractionStore | None = None) -> None:
        self._profiles: dict[str, UserProfile] = {}
        self.interactions = interactions or InteractionStore()

    def get(self, user_id: str) -> dict[str, object]:
        profile = self._profiles.setdefault(user_id, UserProfile(user_id=user_id))
        payload = asdict(profile)
        history = self.interactions.user_history(user_id)
        if history.empty:
            payload["interaction_count"] = 0
            payload["category_affinity"] = {}
            payload["recent_courses"] = []
            return payload

        history = history.sort_values("occurred_at")
        for event_type, field_name in {
            "view": "viewed_courses",
            "click": "clicked_courses",
            "save": "saved_courses",
            "enroll": "enrolled_courses",
            "complete": "completed_courses",
        }.items():
            payload[field_name] = history[history["event_type"].eq(event_type)]["course"].dropna().tail(20).tolist()
        payload["interaction_count"] = int(len(history))
        payload["recent_courses"] = history["course"].dropna().tail(10).tolist()
        payload["category_affinity"] = self._weighted_affinity(history, "category")
        payload["sub_category_affinity"] = self._weighted_affinity(history, "sub_category")
        payload["platform_affinity"] = self._weighted_affinity(history, "platform")
        payload["level_progression"] = self._level_progression(history)
        payload["completion_rate"] = round(
            len(payload["completed_courses"]) / max(len(set(history["course"])), 1),
            3,
        )
        return payload

    def _weighted_affinity(self, history: pd.DataFrame, column: str) -> dict[str, float]:
        now = datetime.now(timezone.utc)
        scores: dict[str, float] = defaultdict(float)
        for row in history.itertuples(index=False):
            label = str(getattr(row, column, "") or "").strip()
            if not label:
                continue
            try:
                occurred_at = datetime.fromisoformat(str(row.occurred_at))
                age_days = max((now - occurred_at).total_seconds() / 86400, 0.0)
            except ValueError:
                age_days = 0.0
            recency = 0.5 ** (age_days / 45.0)
            scores[label] += float(row.weight) * recency
        total = sum(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]
        return {label: round(score / total, 4) for label, score in ranked}

    def _level_progression(self, history: pd.DataFrame) -> str:
        completed = history[history["event_type"].eq("complete")]
        levels = set(completed["level"].dropna().str.lower())
        if "advanced" in levels:
            return "Advanced"
        if "intermediate" in levels or len(completed) >= 3:
            return "Intermediate"
        return "Beginner"

    def score_courses(self, user_id: str, courses: pd.DataFrame) -> pd.Series:
        profile = self.get(user_id)
        if not profile.get("interaction_count"):
            return pd.Series(np.zeros(len(courses)), index=courses.index)

        category_affinity = profile["category_affinity"]
        sub_category_affinity = profile["sub_category_affinity"]
        platform_affinity = profile["platform_affinity"]
        completed = set(profile["completed_courses"])
        progression = str(profile["level_progression"]).lower()
        level_order = {"beginner": 0, "all levels": 1, "intermediate": 2, "advanced": 3}
        progression_order = level_order.get(progression, 0)

        scores = []
        for row in courses.itertuples(index=False):
            score = (
                category_affinity.get(str(row.category), 0.0) * 0.45
                + sub_category_affinity.get(str(row.sub_category), 0.0) * 0.25
                + platform_affinity.get(str(row.site), 0.0) * 0.10
            )
            course_level = level_order.get(str(row.level).lower(), 1)
            if progression_order <= course_level <= progression_order + 1:
                score += 0.20
            if str(row.course) in completed:
                score *= 0.1
            scores.append(min(score, 1.0))
        return pd.Series(scores, index=courses.index)

    def record_event(
        self,
        user_id: str,
        event_type: str,
        course: str,
        category: str = "",
        sub_category: str = "",
        platform: str = "",
        level: str = "",
        course_id: str = "",
        session_id: str = "",
        source: str = "api",
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        profile = self._profiles.setdefault(user_id, UserProfile(user_id=user_id))
        target = {
            "view": profile.viewed_courses,
            "click": profile.clicked_courses,
            "save": profile.clicked_courses,
            "enroll": profile.clicked_courses,
            "complete": profile.completed_courses,
        }.get(event_type)
        if target is not None and course and course not in target:
            target.append(course)
        if course:
            self.interactions.record(
                user_id=user_id,
                event_type=event_type,
                course=course,
                category=category,
                sub_category=sub_category,
                platform=platform,
                level=level,
                course_id=course_id,
                session_id=session_id,
                source=source,
                metadata=metadata,
            )
        return self.get(user_id)


InMemoryProfileStore = UserProfileStore
