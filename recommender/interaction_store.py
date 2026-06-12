from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable
from uuid import uuid4

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_PATH = DATA_DIR / "interactions.db"
VALID_EVENT_TYPES = {"view", "click", "save", "enroll", "complete"}
EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.5,
    "save": 3.5,
    "enroll": 4.0,
    "complete": 5.0,
}


@dataclass(frozen=True)
class InteractionEvent:
    event_id: str
    user_id: str
    course_id: str
    course: str
    event_type: str
    weight: float
    occurred_at: str
    category: str = ""
    sub_category: str = ""
    platform: str = ""
    level: str = ""
    session_id: str = ""
    source: str = "api"
    metadata_json: str = "{}"


class InteractionStore:
    def __init__(self, path: Path = INTERACTIONS_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    course_id TEXT NOT NULL,
                    course TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (
                        event_type IN ('view', 'click', 'save', 'enroll', 'complete')
                    ),
                    weight REAL NOT NULL,
                    occurred_at TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    sub_category TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL DEFAULT '',
                    level TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'api',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_interactions_user_time
                    ON interactions(user_id, occurred_at DESC);
                CREATE INDEX IF NOT EXISTS idx_interactions_course_type
                    ON interactions(course_id, event_type);
                CREATE INDEX IF NOT EXISTS idx_interactions_training
                    ON interactions(user_id, course_id, event_type);
                """
            )

    def record(
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
        occurred_at: str | None = None,
    ) -> InteractionEvent:
        normalized_event = event_type.strip().lower()
        if normalized_event not in VALID_EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {event_type}")
        normalized_user = user_id.strip()
        normalized_course = course.strip()
        if not normalized_user:
            raise ValueError("user_id is required")
        if not normalized_course:
            raise ValueError("course is required")
        normalized_course_id = course_id.strip() or normalized_course
        event = InteractionEvent(
            event_id=str(uuid4()),
            user_id=normalized_user,
            course_id=normalized_course_id,
            course=normalized_course,
            event_type=normalized_event,
            weight=EVENT_WEIGHTS[normalized_event],
            occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
            category=category.strip(),
            sub_category=sub_category.strip(),
            platform=platform.strip(),
            level=level.strip(),
            session_id=session_id.strip(),
            source=source.strip() or "api",
            metadata_json=json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    event_id, user_id, course_id, course, event_type, weight,
                    occurred_at, category, sub_category, platform, level,
                    session_id, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(asdict(event).values()),
            )
        return event

    def iter_events(self, user_id: str | None = None) -> Iterable[InteractionEvent]:
        query = """
            SELECT event_id, user_id, course_id, course, event_type, weight,
                   occurred_at, category, sub_category, platform, level,
                   session_id, source, metadata_json
            FROM interactions
        """
        parameters: tuple[str, ...] = ()
        if user_id:
            query += " WHERE user_id = ?"
            parameters = (user_id,)
        query += " ORDER BY occurred_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [InteractionEvent(**dict(row)) for row in rows]

    def to_frame(self) -> pd.DataFrame:
        rows = [asdict(event) for event in self.iter_events()]
        columns = list(InteractionEvent.__dataclass_fields__)
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def user_history(self, user_id: str) -> pd.DataFrame:
        rows = [asdict(event) for event in self.iter_events(user_id)]
        return pd.DataFrame(rows, columns=list(InteractionEvent.__dataclass_fields__))

    def training_frame(self) -> pd.DataFrame:
        interactions = self.to_frame()
        if interactions.empty:
            return pd.DataFrame(columns=["user_id", "course_id", "course", "implicit_weight"])
        return (
            interactions.groupby(["user_id", "course_id", "course"], as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "implicit_weight"})
        )

    def event_count(self, user_id: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM interactions"
        parameters: tuple[str, ...] = ()
        if user_id:
            query += " WHERE user_id = ?"
            parameters = (user_id,)
        with self._connect() as connection:
            return int(connection.execute(query, parameters).fetchone()[0])
