from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender.collaborative_filtering import ImplicitMatrixFactorization
from recommender.evaluation import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from recommender.explainability import explanation_bullets
from recommender.experiments import run_benchmark, write_reports
from recommender.interaction_store import InteractionStore
from recommender.learning_path import generate_learning_path
from recommender.model_registry import model_metadata
from recommender.ranking import compute_hybrid_score, compute_user_preference_match, get_hybrid_weights
from recommender.resume_analyzer import analyze_skill_gap
from recommender.semantic_search import score_courses
from recommender.user_profile import UserProfileStore


app = FastAPI(
    title="AI Course Discovery & Recommendation Engine",
    description="Multi-factor personalized course discovery API for student goals, level, time, and learning preferences.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = "Online_Courses.csv"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
DEFAULT_LIMIT = 6
MAX_LIMIT = 12
LEVELS = ["Beginner", "Intermediate", "Advanced", "All Levels"]
LEARNING_STYLES = ["Hands-on", "Structured", "Exam-focused", "Fast-track", "Balanced"]
TIME_BUCKETS = ["Flexible", "Weekend only", "5-7 hrs/week", "8-12 hrs/week", "12+ hrs/week"]
COMPLETION_WINDOWS = ["Any", "Within 1 month", "Within 2 months", "Within 3 months", "Within 6 months", "Within 12 months"]
BUDGET_OPTIONS = ["Any", "Free", "Paid"]
DELIVERY_OPTIONS = ["Any", "Project-based", "Theory-based"]
CERTIFICATE_OPTIONS = ["Any", "Yes", "No"]
INTERACTION_STORE = InteractionStore()
PROFILE_STORE = UserProfileStore(INTERACTION_STORE)
COLLABORATIVE_MODEL = ImplicitMatrixFactorization(INTERACTION_STORE)
TEXT_COLUMNS = [
    "course",
    "description",
    "category",
    "sub_category",
    "skills",
    "what_you_learn",
    "prerequisites",
    "course_type",
]
STYLE_KEYWORDS = {
    "Hands-on": ["project", "hands-on", "practical", "build", "lab", "portfolio", "real-world"],
    "Structured": ["specialization", "curriculum", "guided", "foundation", "step-by-step", "module"],
    "Exam-focused": ["exam", "interview", "assessment", "test", "preparation", "gate", "placement"],
    "Fast-track": ["introduction", "crash", "quick", "fundamentals", "basics", "starter"],
    "Balanced": ["overview", "comprehensive", "applied", "practice", "concept", "project"],
}
GOAL_ALIASES = {
    " ml ": " machine learning ",
    " ai ": " artificial intelligence ",
    "fullstack": "full stack",
    "front end": "frontend",
    "back end": "backend",
    "mern": "mern stack",
    "dsa": "data structures and algorithms",
}
GOAL_EXPANSIONS = {
    "full stack": [
        "web development",
        "frontend",
        "backend",
        "html",
        "css",
        "javascript",
        "react",
        "node",
        "express",
        "api",
        "sql",
        "database",
    ],
    "frontend": ["web development", "html", "css", "javascript", "react", "ui", "responsive web design"],
    "backend": ["server", "api", "database", "sql", "node", "django", "flask", "express", "spring", "java", "fastapi"],
    "java backend": ["java", "spring", "spring boot", "backend", "api", "microservices", "server", "database"],
    "python backend": ["python", "django", "flask", "fastapi", "backend", "api", "server", "database"],
    "mern stack": ["mongodb", "express", "react", "node", "javascript", "web development", "full stack"],
    "blockchain": ["blockchain", "web3", "smart contract", "ethereum", "solidity", "distributed ledger"],
    "network engineer": ["networking", "computer networks", "network security", "routing", "switching", "infrastructure"],
    "database administrator": ["database", "sql", "relational databases", "rdbms", "data management", "mysql", "postgresql"],
    "data structures and algorithms": ["algorithms", "data structures", "problem solving", "computational thinking", "coding interview"],
    "software developer": ["software development", "programming", "application development", "object oriented programming"],
    "web developer": ["web development", "html", "css", "javascript", "frontend", "backend", "react", "node"],
}
GOAL_PROFILES = [
    {
        "name": "full_stack",
        "triggers": ["full stack", "web developer", "frontend", "backend"],
        "keywords": ["web development", "html", "css", "javascript", "react", "node", "express", "django", "sql", "api"],
        "categories": ["computer science", "information technology"],
        "subcategories": ["mobile and web development", "software development", "cloud computing"],
    },
    {
        "name": "data_science",
        "triggers": ["data science", "data scientist", "data analyst", "machine learning", "ml engineer"],
        "keywords": ["python", "machine learning", "data analysis", "statistics", "sql", "pandas", "numpy", "visualization"],
        "categories": ["data science", "computer science"],
        "subcategories": ["data analysis", "machine learning", "probability and statistics"],
    },
    {
        "name": "cybersecurity",
        "triggers": ["cybersecurity", "cyber security", "security analyst", "ethical hacker", "soc analyst"],
        "keywords": ["cybersecurity", "network security", "cryptography", "security", "threat", "incident response"],
        "categories": ["information technology", "computer science"],
        "subcategories": ["security", "computer security and networks", "cloud computing"],
    },
    {
        "name": "ui_ux",
        "triggers": ["ui ux", "ux", "ui design", "ux design", "product design", "designer"],
        "keywords": ["ux", "ui", "wireframe", "prototype", "figma", "design thinking", "user research", "responsive design"],
        "categories": ["computer science"],
        "subcategories": ["design and product", "mobile and web development"],
    },
    {
        "name": "devops",
        "triggers": ["devops", "site reliability", "sre", "platform engineer", "cloud engineer"],
        "keywords": ["devops", "docker", "kubernetes", "ci/cd", "cloud computing", "terraform", "linux", "monitoring", "deployment"],
        "categories": ["information technology", "computer science"],
        "subcategories": ["cloud computing", "software development", "data management"],
    },
    {
        "name": "python_dev",
        "triggers": ["python developer", "python engineer", "django developer", "flask developer", "python backend"],
        "keywords": ["python", "django", "flask", "fastapi", "api", "software development", "object oriented programming", "database"],
        "categories": ["computer science", "information technology"],
        "subcategories": ["software development", "mobile and web development"],
    },
    {
        "name": "java_backend",
        "triggers": ["java backend", "spring boot", "java developer", "backend java"],
        "keywords": ["java", "spring", "spring boot", "backend", "api", "microservices", "database"],
        "categories": ["computer science", "information technology"],
        "subcategories": ["software development", "mobile and web development"],
    },
    {
        "name": "mern_stack",
        "triggers": ["mern", "mern stack", "mongodb express react node"],
        "keywords": ["mongodb", "express", "react", "node", "javascript", "full stack", "web development"],
        "categories": ["computer science", "information technology"],
        "subcategories": ["mobile and web development", "software development"],
    },
    {
        "name": "blockchain",
        "triggers": ["blockchain", "web3", "smart contract", "solidity"],
        "keywords": ["blockchain", "web3", "smart contract", "ethereum", "solidity", "distributed ledger"],
        "categories": ["computer science", "information technology"],
        "subcategories": ["software development", "data management"],
    },
    {
        "name": "networking",
        "triggers": ["network engineer", "networking", "computer networks", "network administrator"],
        "keywords": ["networking", "computer networks", "routing", "switching", "infrastructure", "network security"],
        "categories": ["information technology", "computer science"],
        "subcategories": ["networking", "computer security and networks", "security"],
    },
    {
        "name": "database_admin",
        "triggers": ["database administrator", "dba", "database admin", "sql developer"],
        "keywords": ["database", "sql", "rdbms", "relational databases", "data management", "mysql", "postgresql"],
        "categories": ["information technology", "data science", "computer science"],
        "subcategories": ["data management", "data analysis", "software development"],
    },
    {
        "name": "algorithms_dsa",
        "triggers": ["algorithms", "data structures", "dsa", "data structures and algorithms", "coding interview"],
        "keywords": ["algorithms", "data structures", "problem solving", "computational thinking", "programming"],
        "categories": ["computer science"],
        "subcategories": ["algorithms", "software development", "math and logic"],
    },
]


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _normalize_label(value: str) -> str:
    return _clean_text(value).lower()


def _normalize_language(value: object) -> str:
    normalized = _normalize_label(str(value))
    if not normalized:
        return ""
    aliases = {
        "english us": "english",
        "english uk": "english",
        "en": "english",
    }
    return aliases.get(normalized, normalized)


def _parse_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    match = re.search(r"\d+(\.\d+)?", str(value))
    return float(match.group()) if match else 0.0


def _parse_int(value: object) -> int:
    if pd.isna(value):
        return 0
    cleaned = re.sub(r"[^\d]", "", str(value))
    return int(cleaned) if cleaned else 0


def _safe_log_score(value: int, scale: int) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(scale), 1.0)


def _format_level(value: object) -> str:
    text = _clean_text(value)
    return text if text else "All Levels"


def _infer_level(raw_level: str, course: str, description: str, course_type: str) -> str:
    explicit = _format_level(raw_level)
    if explicit != "All Levels":
        return explicit

    text = " ".join([course, description, course_type]).lower()

    advanced_keywords = [
        "advanced",
        "expert",
        "professional certificate",
        "production",
        "mlops",
        "engineering for production",
        "optimization",
    ]
    intermediate_keywords = [
        "intermediate",
        "applied",
        "case study",
        "specialization",
        "project",
        "practical",
        "for business",
    ]
    beginner_keywords = [
        "beginner",
        "beginners",
        "introduction",
        "intro",
        "fundamentals",
        "foundation",
        "foundations",
        "basics",
        "starter",
        "for everyone",
        "101",
    ]

    if any(keyword in text for keyword in advanced_keywords):
        return "Advanced"
    if any(keyword in text for keyword in beginner_keywords):
        return "Beginner"
    if any(keyword in text for keyword in intermediate_keywords):
        return "Intermediate"
    return "All Levels"


def _parse_duration_hours(text: str) -> float:
    value = _clean_text(text).lower()
    if not value:
        return 0.0

    paced_match = re.search(
        r"(\d+(\.\d+)?)\s*(week|weeks|month|months)\s+at\s+(\d+(\.\d+)?)\s*hours?\s+a\s+week",
        value,
    )
    if paced_match:
        span_amount = float(paced_match.group(1))
        span_unit = paced_match.group(3)
        weekly_hours = float(paced_match.group(4))
        total_weeks = span_amount * 4.0 if "month" in span_unit else span_amount
        return weekly_hours * total_weeks

    match = re.search(r"(\d+(\.\d+)?)", value)
    amount = float(match.group(1)) if match else 0.0

    if "hour" in value:
        return amount
    if "day" in value:
        return amount * 2.5
    if "week" in value:
        return amount * 5.0
    if "month" in value:
        return amount * 20.0
    return 0.0


def _duration_bucket(hours: float) -> str:
    if hours <= 0:
        return "Flexible pace"
    if hours <= 12:
        return "Short"
    if hours <= 35:
        return "Medium"
    return "Long"


def _time_target_hours(time_commitment: str) -> tuple[float, float]:
    mapping = {
        "Flexible": (20.0, 999.0),
        "Weekend only": (10.0, 18.0),
        "5-7 hrs/week": (18.0, 30.0),
        "8-12 hrs/week": (25.0, 45.0),
        "12+ hrs/week": (35.0, 999.0),
    }
    return mapping.get(time_commitment, (20.0, 999.0))


def _time_fit_score(duration_hours: float, time_commitment: str) -> float:
    if duration_hours <= 0 or time_commitment == "Flexible":
        return 0.7

    minimum, maximum = _time_target_hours(time_commitment)
    if minimum <= duration_hours <= maximum:
        return 1.0
    if duration_hours < minimum:
        return max(0.65, duration_hours / minimum)
    overflow = duration_hours - maximum
    return max(0.2, 1 - (overflow / max(maximum, 1.0)))


def _minimum_time_fit(time_commitment: str) -> float:
    thresholds = {
        "Flexible": 0.0,
        "Weekend only": 0.95,
        "5-7 hrs/week": 0.8,
        "8-12 hrs/week": 0.75,
        "12+ hrs/week": 0.6,
    }
    return thresholds.get(time_commitment, 0.7)


def _time_commitment_weekly_hours(time_commitment: str) -> float:
    mapping = {
        "Flexible": 6.0,
        "Weekend only": 4.0,
        "5-7 hrs/week": 6.0,
        "8-12 hrs/week": 10.0,
        "12+ hrs/week": 14.0,
    }
    return mapping.get(time_commitment, 6.0)


def _completion_window_limit(completion_months: str) -> float:
    mapping = {
        "Within 1 month": 1.0,
        "Within 2 months": 2.0,
        "Within 3 months": 3.0,
        "Within 6 months": 6.0,
        "Within 12 months": 12.0,
    }
    return mapping.get(completion_months, 0.0)


def _estimated_completion_months(duration_hours: float, time_commitment: str) -> float:
    if duration_hours <= 0:
        return 0.0
    weekly_hours = _time_commitment_weekly_hours(time_commitment)
    if weekly_hours <= 0:
        return 0.0
    return duration_hours / weekly_hours / 4.0


def _completion_fit_score(duration_hours: float, time_commitment: str, completion_months: str) -> float:
    if completion_months == "Any":
        return 0.7

    estimated_months = _estimated_completion_months(duration_hours, time_commitment)
    if estimated_months <= 0:
        return 0.65

    target_months = _completion_window_limit(completion_months)
    if target_months <= 0:
        return 0.7
    if estimated_months <= target_months:
        return 1.0
    if estimated_months <= target_months * 1.25:
        return 0.82
    if estimated_months <= target_months * 1.6:
        return 0.58
    return 0.22


def _minimum_completion_fit(completion_months: str) -> float:
    thresholds = {
        "Any": 0.0,
        "Within 1 month": 0.95,
        "Within 2 months": 0.88,
        "Within 3 months": 0.82,
        "Within 6 months": 0.72,
        "Within 12 months": 0.6,
    }
    return thresholds.get(completion_months, 0.0)


def _format_completion_label(estimated_months: float) -> str:
    if estimated_months <= 0:
        return "Depends on pace"
    if estimated_months < 1:
        weeks = max(1, round(estimated_months * 4))
        return f"About {weeks} week{'s' if weeks != 1 else ''}"
    rounded = round(estimated_months, 1)
    if abs(rounded - round(rounded)) < 0.05:
        rounded = int(round(rounded))
    return f"About {rounded} month{'s' if rounded != 1 else ''}"


def _detect_budget_type(program_type: str, price: str, premium_course: str) -> str:
    program_text = _normalize_label(program_type)
    premium_text = _normalize_label(premium_course)
    price_value = _parse_float(price)

    if "free" in program_text:
        return "Free"
    if price_value > 0:
        return "Paid"
    if premium_text.startswith("$") or "pay $" in premium_text or "premium course" in premium_text:
        return "Paid"
    return "Free"


def _budget_fit_score(budget_type: str, budget_preference: str) -> float:
    if budget_preference == "Any":
        return 0.7
    if budget_type == budget_preference:
        return 1.0
    return 0.2


def _detect_delivery_mode(course_type: str, style_text: str) -> str:
    course_type_text = _normalize_label(course_type)
    style_text = style_text.lower()
    strong_project_keywords = ["hands-on", "portfolio", "real-world", "guided project", "capstone", "lab"]
    project_term_hits = sum(style_text.count(term) for term in ["project", "projects"])

    if course_type_text == "project":
        return "Project-based"
    if any(keyword in style_text for keyword in strong_project_keywords):
        return "Project-based"
    if project_term_hits >= 2:
        return "Project-based"
    return "Theory-based"


def _delivery_fit_score(delivery_mode: str, delivery_preference: str) -> float:
    if delivery_preference == "Any":
        return 0.7
    if delivery_mode == delivery_preference:
        return 1.0
    return 0.35


def _detect_certificate_availability(course_type: str, program_type: str, premium_course: str, whats_include: str, description: str) -> bool:
    certificate_text = " ".join([course_type, program_type, premium_course, whats_include, description]).lower()
    keywords = ["certificate", "certificate of achievement", "professional certificate", "digital certificate", "award"]
    return any(keyword in certificate_text for keyword in keywords)


def _certificate_fit_score(has_certificate: bool, certificate_preference: str) -> float:
    if certificate_preference == "Any":
        return 0.7
    if certificate_preference == "Yes":
        return 1.0 if has_certificate else 0.2
    if certificate_preference == "No":
        return 1.0 if not has_certificate else 0.55
    return 0.7


def _classify_style_score(text: str, learning_style: str) -> float:
    if learning_style not in STYLE_KEYWORDS:
        return 0.7
    lowered = text.lower()
    matches = sum(keyword in lowered for keyword in STYLE_KEYWORDS[learning_style])
    return min(1.0, 0.35 + matches * 0.2)


def _normalize_goal_text(goal: str) -> str:
    normalized = f" {_normalize_label(goal)} "
    for source, target in GOAL_ALIASES.items():
        normalized = normalized.replace(source, target)
    return normalized.strip()


def _expand_goal_text(goal: str) -> str:
    normalized = _normalize_goal_text(goal)
    expanded_terms = [normalized]

    for phrase, related_terms in GOAL_EXPANSIONS.items():
        if phrase in normalized:
            expanded_terms.extend(related_terms)

    if "developer" in normalized and "web" not in normalized and "full stack" not in normalized:
        expanded_terms.extend(["software development", "programming", "application development"])

    return " ".join(dict.fromkeys(term for term in expanded_terms if term))


def _keyword_goal_score(goal: str, course_text: str) -> float:
    normalized_goal = _normalize_goal_text(goal)
    if not normalized_goal:
        return 0.0

    expanded_terms = _expand_goal_text(goal).split()
    goal_keywords = [
        token for token in dict.fromkeys(expanded_terms)
        if len(token) > 2 and token not in {"become", "want", "developer"}
    ]
    if not goal_keywords:
        return 0.0

    lowered_text = course_text.lower()
    matches = sum(1 for keyword in goal_keywords if keyword in lowered_text)
    return min(1.0, matches / max(len(goal_keywords) * 0.45, 1.0))


def _matching_goal_profiles(goal: str) -> list[dict[str, list[str] | str]]:
    normalized_goal = _normalize_goal_text(goal)
    if not normalized_goal:
        return []

    matched_profiles = []
    for profile in GOAL_PROFILES:
        if any(trigger in normalized_goal for trigger in profile["triggers"]):
            matched_profiles.append(profile)
    return matched_profiles


def _profile_goal_score(goal: str, course_text: str, category: str, sub_category: str) -> float:
    matched_profiles = _matching_goal_profiles(goal)
    if not matched_profiles:
        return 0.0

    lowered_text = course_text.lower()
    normalized_category = _normalize_label(category)
    normalized_sub_category = _normalize_label(sub_category)
    best_score = 0.0

    for profile in matched_profiles:
        keyword_hits = sum(1 for keyword in profile["keywords"] if keyword in lowered_text)
        category_match = normalized_category in profile["categories"]
        sub_category_match = normalized_sub_category in profile["subcategories"]

        score = min(1.0, keyword_hits / max(len(profile["keywords"]) * 0.3, 1.0))
        if category_match:
            score += 0.18
        if sub_category_match:
            score += 0.24
        best_score = max(best_score, min(score, 1.0))

    return best_score


def _build_reason(row: pd.Series, profile: dict[str, str]) -> str:
    reasons = []

    if row["goal_fit"] >= 0.45:
        reasons.append("strong goal alignment")
    elif row["goal_fit"] > 0.2:
        reasons.append("clear topic relevance")

    if row["level_fit"] >= 1.0:
        reasons.append(f"well matched to a {row['level']} learner")

    if row["time_fit"] >= 0.95:
        reasons.append(f"manageable for {profile['time_commitment'].lower()}")

    if row["completion_fit"] >= 0.95 and profile["completion_months"] != "Any":
        reasons.append(f"supports your {profile['completion_months'].lower()} target")

    if row["budget_fit"] >= 0.95 and profile["budget_preference"] != "Any":
        reasons.append(f"matches your {profile['budget_preference'].lower()} budget preference")

    if row["delivery_fit"] >= 0.95 and profile["delivery_preference"] != "Any":
        reasons.append(f"leans {profile['delivery_preference'].lower()}")

    if row["certificate_fit"] >= 0.95 and profile["certificate_preference"] == "Yes":
        reasons.append("includes certificate support")

    if row["style_fit"] >= 0.9:
        reasons.append(f"fits a {profile['learning_style'].lower()} learning style")

    if row["rating_value"] >= 4.5:
        reasons.append(f"strong peer signal ({row['rating_value']:.1f}/5 rating)")

    if row["platform_fit"] >= 1.0 and row["site"]:
        reasons.append(f"available on your preferred platform, {row['site']}")

    return ". ".join(reasons[:4]).capitalize() + "." if reasons else "Solid fit based on your current profile."


def _peer_feedback_summary(row: pd.Series) -> str:
    parts = []
    if row["rating_value"] > 0:
        parts.append(f"Rated {row['rating_value']:.1f}/5")
    if row["social_proof"] > 0:
        parts.append(f"backed by {row['social_proof']:,}+ learner interactions")
    if row["course_type"]:
        parts.append(f"offered as a {row['course_type'].lower()}")
    return ". ".join(parts) + "." if parts else "Peer feedback data is limited for this course."


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@lru_cache(maxsize=1)
def _load_assets() -> tuple[pd.DataFrame, TfidfVectorizer, object, dict[str, list[str]]]:
    courses = pd.read_csv(DATASET_PATH)
    courses = courses.rename(
        columns={
            "Title": "course",
            "Short Intro": "description",
            "Level": "level",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Skills": "skills",
            "What you learn": "what_you_learn",
            "Prequisites": "prerequisites",
            "Rating": "rating",
            "URL": "url",
            "Site": "site",
            "Duration": "duration",
            "Language": "language",
            "Course Type": "course_type",
            "Number of Reviews": "review_count",
            "Number of ratings": "rating_count",
            "Number of viewers": "viewer_count",
            "Instructors": "instructors",
            "Price": "price",
            "Program Type": "program_type",
            "Premium course": "premium_course",
            "What's include": "whats_include",
        }
    )

    required_columns = [
        "course",
        "description",
        "level",
        "category",
        "sub_category",
        "skills",
        "what_you_learn",
        "prerequisites",
        "rating",
        "url",
        "site",
        "duration",
        "language",
        "course_type",
        "review_count",
        "rating_count",
        "viewer_count",
        "instructors",
        "price",
        "program_type",
        "premium_course",
        "whats_include",
    ]
    for column in required_columns:
        if column not in courses.columns:
            courses[column] = ""
        courses[column] = courses[column].apply(_clean_text)

    courses = courses.dropna(subset=["course", "description"]).copy()
    courses = courses.drop_duplicates(subset=["course", "site", "url"], keep="first").reset_index(drop=True)
    courses["course_id"] = courses.apply(
        lambda row: hashlib.sha256(
            f"{row['course']}|{row['site']}|{row['url']}".encode("utf-8")
        ).hexdigest()[:20],
        axis=1,
    )
    courses["level"] = courses.apply(
        lambda row: _infer_level(
            row["level"],
            row["course"],
            row["description"],
            row["course_type"],
        ),
        axis=1,
    )
    courses["rating_value"] = courses["rating"].apply(_parse_float)
    courses["review_value"] = courses["review_count"].apply(_parse_int)
    courses["rating_count_value"] = courses["rating_count"].apply(_parse_int)
    courses["viewer_value"] = courses["viewer_count"].apply(_parse_int)
    courses["social_proof"] = (
        courses["review_value"] + courses["rating_count_value"] + courses["viewer_value"]
    )
    courses["popularity_score"] = courses["social_proof"].apply(lambda value: _safe_log_score(value, 500000))
    courses["duration_hours"] = courses["duration"].apply(_parse_duration_hours)
    courses["duration_bucket"] = courses["duration_hours"].apply(_duration_bucket)
    courses["language_normalized"] = courses["language"].apply(_normalize_language)
    courses["style_text"] = courses[TEXT_COLUMNS].agg(" ".join, axis=1).str.lower()
    courses["budget_type"] = courses.apply(
        lambda row: _detect_budget_type(row["program_type"], row["price"], row["premium_course"]),
        axis=1,
    )
    courses["delivery_mode"] = courses.apply(
        lambda row: _detect_delivery_mode(row["course_type"], row["style_text"]),
        axis=1,
    )
    courses["has_certificate"] = courses.apply(
        lambda row: _detect_certificate_availability(
            row["course_type"],
            row["program_type"],
            row["premium_course"],
            row["whats_include"],
            row["description"],
        ),
        axis=1,
    )
    courses["goal_text"] = (
        courses[["course", "description", "category", "sub_category", "skills", "what_you_learn", "site"]]
        .agg(" ".join, axis=1)
        .str.lower()
    )
    courses["content"] = (
        courses[TEXT_COLUMNS + ["site", "language", "instructors"]].agg(" ".join, axis=1).str.lower()
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=35000)
    tfidf_matrix = vectorizer.fit_transform(courses["content"])

    metadata = {
        "platforms": sorted(platform for platform in courses["site"].dropna().unique() if platform),
        "categories": sorted(category for category in courses["category"].dropna().unique() if category)[:25],
        "languages": sorted(language for language in courses["language"].dropna().unique() if language),
        "levels": LEVELS,
        "learning_styles": LEARNING_STYLES,
        "time_commitments": TIME_BUCKETS,
        "completion_windows": COMPLETION_WINDOWS,
        "budget_options": BUDGET_OPTIONS,
        "delivery_options": DELIVERY_OPTIONS,
        "certificate_options": CERTIFICATE_OPTIONS,
    }
    return courses, vectorizer, tfidf_matrix, metadata


@lru_cache(maxsize=256)
def _recommend_cached(
    goal: str,
    level: str,
    learning_style: str,
    time_commitment: str,
    completion_months: str,
    budget_preference: str,
    delivery_preference: str,
    certificate_preference: str,
    preferred_platform: str,
    preferred_language: str,
    category: str,
    limit: int,
    user_id: str = "",
    interaction_revision: int = 0,
) -> dict[str, object]:
    courses, vectorizer, tfidf_matrix, _ = _load_assets()
    temp_courses = courses.copy()

    goal_text = _normalize_goal_text(goal)
    expanded_goal_text = _expand_goal_text(goal)
    level_text = level.strip().lower()
    platform_text = preferred_platform.strip().lower()
    language_text = _normalize_language(preferred_language)
    category_text = category.strip().lower()
    budget_text = budget_preference.strip()
    delivery_text = delivery_preference.strip()
    certificate_text = certificate_preference.strip()

    semantic_backend = "tfidf"
    if expanded_goal_text:
        semantic_result = score_courses(courses, expanded_goal_text)
        if semantic_result.scores is not None:
            temp_courses["semantic_score"] = semantic_result.scores
            semantic_backend = semantic_result.backend
        else:
            goal_vector = vectorizer.transform([expanded_goal_text])
            temp_courses["semantic_score"] = cosine_similarity(goal_vector, tfidf_matrix).flatten()
    else:
        temp_courses["semantic_score"] = 0.0
    temp_courses["keyword_goal_score"] = temp_courses["goal_text"].apply(
        lambda text: _keyword_goal_score(goal_text, text)
    )
    temp_courses["profile_goal_score"] = temp_courses.apply(
        lambda row: _profile_goal_score(goal_text, row["goal_text"], row["category"], row["sub_category"]),
        axis=1,
    )
    temp_courses["goal_fit"] = (
        temp_courses["semantic_score"] * 0.4
        + temp_courses["keyword_goal_score"] * 0.3
        + temp_courses["profile_goal_score"] * 0.3
    )

    temp_courses["level_fit"] = 0.65
    if level_text and level_text != "all levels":
        exact_match = temp_courses["level"].str.lower().eq(level_text)
        all_levels = temp_courses["level"].str.lower().eq("all levels")
        temp_courses.loc[all_levels, "level_fit"] = 0.82
        temp_courses.loc[exact_match, "level_fit"] = 1.0

    temp_courses["platform_fit"] = 0.75
    if platform_text and platform_text != "any":
        matched_platform = temp_courses["site"].str.lower().eq(platform_text)
        temp_courses.loc[matched_platform, "platform_fit"] = 1.0
        temp_courses.loc[~matched_platform, "platform_fit"] = 0.55

    temp_courses["language_fit"] = 0.75
    if language_text and language_text != "any":
        matched_language = temp_courses["language_normalized"].str.contains(language_text, regex=False)
        temp_courses.loc[matched_language, "language_fit"] = 1.0
        temp_courses.loc[~matched_language, "language_fit"] = 0.2

    temp_courses["category_fit"] = 0.75
    if category_text and category_text != "any":
        matched_category = temp_courses["category"].str.lower().eq(category_text)
        matched_sub_category = temp_courses["sub_category"].str.lower().eq(category_text)
        temp_courses.loc[matched_category | matched_sub_category, "category_fit"] = 1.0
        temp_courses.loc[~(matched_category | matched_sub_category), "category_fit"] = 0.45

    temp_courses["time_fit"] = temp_courses["duration_hours"].apply(
        lambda value: _time_fit_score(value, time_commitment)
    )
    temp_courses["estimated_completion_months"] = temp_courses["duration_hours"].apply(
        lambda value: _estimated_completion_months(value, time_commitment)
    )
    temp_courses["completion_fit"] = temp_courses["duration_hours"].apply(
        lambda value: _completion_fit_score(value, time_commitment, completion_months)
    )
    temp_courses["style_fit"] = temp_courses["style_text"].apply(
        lambda text: _classify_style_score(text, learning_style)
    )
    temp_courses["budget_fit"] = temp_courses["budget_type"].apply(
        lambda value: _budget_fit_score(value, budget_text)
    )
    temp_courses["delivery_fit"] = temp_courses["delivery_mode"].apply(
        lambda value: _delivery_fit_score(value, delivery_text)
    )
    temp_courses["certificate_fit"] = temp_courses["has_certificate"].apply(
        lambda value: _certificate_fit_score(bool(value), certificate_text)
    )
    temp_courses["quality_score"] = temp_courses["rating_value"].apply(lambda value: min(value / 5.0, 1.0))

    temp_courses["user_preference_match"] = compute_user_preference_match(temp_courses)
    if user_id:
        temp_courses["behavioral_profile_score"] = PROFILE_STORE.score_courses(user_id, temp_courses)
        collaborative_result = COLLABORATIVE_MODEL.score(user_id, temp_courses)
        temp_courses["collaborative_score"] = collaborative_result.scores
        behavioral_profile = PROFILE_STORE.get(user_id)
    else:
        temp_courses["behavioral_profile_score"] = 0.0
        temp_courses["collaborative_score"] = 0.0
        collaborative_result = None
        behavioral_profile = {}
    temp_courses["final_score"] = compute_hybrid_score(temp_courses)

    filtered = temp_courses.copy()
    if platform_text and platform_text != "any":
        strict_platform = filtered[filtered["site"].str.lower().eq(platform_text)]
        if len(strict_platform) >= limit:
            filtered = strict_platform
    if level_text and level_text != "all levels":
        exact_level_filtered = filtered[filtered["level"].str.lower().eq(level_text)]
        fallback_level_filtered = filtered[
            filtered["level"].str.lower().eq(level_text) | filtered["level"].str.lower().eq("all levels")
        ]
        if len(exact_level_filtered) >= max(3, limit // 2):
            filtered = exact_level_filtered
        elif not fallback_level_filtered.empty:
            filtered = fallback_level_filtered
    if language_text and language_text != "any":
        strict_language = filtered[filtered["language_normalized"].str.contains(language_text, regex=False)]
        if len(strict_language) >= max(3, limit // 2):
            filtered = strict_language
    if time_commitment != "Flexible":
        strict_time_matches = filtered[filtered["time_fit"] >= _minimum_time_fit(time_commitment)]
        if len(strict_time_matches) >= max(3, limit // 2):
            filtered = strict_time_matches
    if completion_months != "Any":
        strict_completion_matches = filtered[filtered["completion_fit"] >= _minimum_completion_fit(completion_months)]
        if len(strict_completion_matches) >= max(3, limit // 2):
            filtered = strict_completion_matches
    if budget_text != "Any":
        strict_budget = filtered[filtered["budget_type"].eq(budget_text)]
        if not strict_budget.empty:
            filtered = strict_budget
    if delivery_text != "Any":
        strict_delivery = filtered[filtered["delivery_mode"].eq(delivery_text)]
        if not strict_delivery.empty:
            filtered = strict_delivery
    if certificate_text == "Yes":
        strict_certificate = filtered[filtered["has_certificate"]]
        if not strict_certificate.empty:
            filtered = strict_certificate

    if goal_text:
        relevant_matches = filtered[filtered["goal_fit"] >= 0.18]
        if len(relevant_matches) >= max(3, limit):
            filtered = relevant_matches
        filtered = filtered.sort_values(
            by=["final_score", "goal_fit", "time_fit", "semantic_score", "quality_score", "popularity_score"],
            ascending=False,
        )
    else:
        filtered = filtered.sort_values(
            by=["quality_score", "popularity_score", "final_score"],
            ascending=False,
        )

    filtered = (
        filtered
        .drop_duplicates(subset=["course", "platform"] if "platform" in filtered.columns else ["course", "site"])
        .head(limit)
    )
    profile = {
        "goal": goal.strip(),
        "level": level,
        "learning_style": learning_style,
        "time_commitment": time_commitment,
        "completion_months": completion_months,
        "budget_preference": budget_preference,
        "delivery_preference": delivery_preference,
        "certificate_preference": certificate_preference,
        "preferred_platform": preferred_platform,
        "preferred_language": preferred_language,
        "category": category,
    }

    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append(
            {
                "course": row["course"],
                "course_id": row["course_id"],
                "level": row["level"],
                "score": round(float(row["final_score"]), 2),
                "goal_score": round(float(row["goal_fit"]), 2),
                "reason": _build_reason(row, profile),
                "explanation_bullets": explanation_bullets(row, profile, semantic_backend),
                "description": row["description"][:260],
                "peer_feedback_summary": _peer_feedback_summary(row),
                "difficulty": row["level"],
                "time_estimate": row["duration"] or row["duration_bucket"],
                "time_bucket": row["duration_bucket"],
                "estimated_completion": _format_completion_label(float(row["estimated_completion_months"])),
                "learning_style_fit": learning_style,
                "budget_type": row["budget_type"],
                "delivery_mode": row["delivery_mode"],
                "certificate_available": bool(row["has_certificate"]),
                "category": row["category"],
                "sub_category": row["sub_category"],
                "skills": row["skills"],
                "rating": row["rating"] or "N/A",
                "platform": row["site"],
                "language": row["language"],
                "course_type": row["course_type"],
                "instructors": row["instructors"],
                "url": row["url"],
                "comparison": {
                    "Goal alignment": round(float(row["goal_fit"]), 2),
                    "Level fit": round(float(row["level_fit"]), 2),
                    "Time fit": round(float(row["time_fit"]), 2),
                    "Completion fit": round(float(row["completion_fit"]), 2),
                    "Budget fit": round(float(row["budget_fit"]), 2),
                    "Delivery fit": round(float(row["delivery_fit"]), 2),
                    "Certificate fit": round(float(row["certificate_fit"]), 2),
                    "Learning style fit": round(float(row["style_fit"]), 2),
                    "User preference match": round(float(row["user_preference_match"]), 2),
                    "Behavioral profile": round(float(row["behavioral_profile_score"]), 2),
                    "Collaborative filtering": round(float(row["collaborative_score"]), 2),
                    "Semantic similarity": round(float(row["semantic_score"]), 2),
                    "Peer signal": round(float((row["quality_score"] * 0.7) + (row["popularity_score"] * 0.3)), 2),
                },
            }
        )

    top_three = recommendations[:3]
    comparison_table = []
    for course in top_three:
        comparison_table.append(
            {
                "Course": course["course"],
                "Platform": course["platform"],
                "Difficulty": course["difficulty"],
                "Time": course["time_estimate"],
                "Finish by": course["estimated_completion"],
                "Budget": course["budget_type"],
                "Delivery": course["delivery_mode"],
                "Certificate": "Yes" if course["certificate_available"] else "No",
                "Rating": course["rating"],
                "Why it fits": course["reason"],
            }
        )

    refinement_prompts = [
        "Need something free? Switch budget to Free.",
        "Want more building and practice? Set project style to Project-based.",
        "Need proof of completion? Set certificate needed to Yes.",
    ]

    return {
        "profile": profile,
        "count": len(recommendations),
        "recommendations": recommendations,
        "comparison_table": comparison_table,
        "coach_message": (
            f"These courses are ranked for a {level.lower()} learner targeting '{goal.strip() or 'career growth'}', "
            f"with a {learning_style.lower()} preference, {time_commitment.lower()} availability, "
            f"a {completion_months.lower()} finish target, {budget_preference.lower()} budget preference, "
            f"and {delivery_preference.lower()} delivery preference."
        ),
        "ranking_model": {
            "type": "hybrid_collaborative" if user_id else "hybrid",
            "semantic_backend": semantic_backend,
            "collaborative_backend": collaborative_result.backend if collaborative_result else "disabled",
            "weights": get_hybrid_weights(),
        },
        "behavioral_profile": behavioral_profile,
        "model_versions": model_metadata(),
        "refinement_prompts": refinement_prompts,
    }


@app.get("/api")
def api_home() -> dict[str, object]:
    courses, _, _, metadata = _load_assets()
    return {
        "message": "AI Course Discovery & Recommendation Engine is running.",
        "total_courses": int(len(courses)),
        "platforms": metadata["platforms"][:6],
        "levels": metadata["levels"],
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/metadata")
def metadata() -> dict[str, object]:
    courses, _, _, metadata = _load_assets()
    return {
        "total_courses": int(len(courses)),
        "platforms": ["Any"] + metadata["platforms"],
        "categories": ["Any"] + metadata["categories"],
        "languages": ["Any"] + metadata["languages"],
        "levels": metadata["levels"],
        "learning_styles": metadata["learning_styles"],
        "time_commitments": metadata["time_commitments"],
        "completion_windows": metadata["completion_windows"],
        "budget_options": metadata["budget_options"],
        "delivery_options": metadata["delivery_options"],
        "certificate_options": metadata["certificate_options"],
    }


@app.get("/recommend")
def recommend(
    goal: str = Query(default="", max_length=300),
    level: str = Query(default="Beginner"),
    learning_style: str = Query(default="Balanced"),
    time_commitment: str = Query(default="5-7 hrs/week"),
    completion_months: str = Query(default="Any"),
    budget_preference: str = Query(default="Any"),
    delivery_preference: str = Query(default="Any"),
    certificate_preference: str = Query(default="Any"),
    preferred_platform: str = Query(default="Any"),
    preferred_language: str = Query(default="Any"),
    category: str = Query(default="Any"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    user_id: str = Query(default="", max_length=120),
) -> dict[str, object]:
    interaction_revision = INTERACTION_STORE.event_count(user_id) if user_id else 0
    return _recommend_cached(
        goal=goal,
        level=level,
        learning_style=learning_style,
        time_commitment=time_commitment,
        completion_months=completion_months,
        budget_preference=budget_preference,
        delivery_preference=delivery_preference,
        certificate_preference=certificate_preference,
        preferred_platform=preferred_platform,
        preferred_language=preferred_language,
        category=category,
        limit=limit,
        user_id=user_id,
        interaction_revision=interaction_revision,
    )


@app.get("/learning-path")
def learning_path(goal: str = Query(default="Machine Learning Engineer", max_length=120)) -> dict[str, object]:
    path = generate_learning_path(goal)
    return {
        "goal": goal,
        "path": path,
        "steps": [{"order": index + 1, "skill": skill} for index, skill in enumerate(path)],
    }


@app.post("/resume/analyze")
async def resume_analyze(
    target_role: str = Form(default="Machine Learning Engineer"),
    resume_text: str = Form(default=""),
    resume_file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    text = resume_text
    if resume_file is not None:
        content = await resume_file.read()
        text = content.decode("utf-8", errors="ignore")

    gap = analyze_skill_gap(text, target_role)
    recommendations = _recommend_cached(
        goal=str(gap["recommendation_goal"]),
        level="Beginner",
        learning_style="Balanced",
        time_commitment="5-7 hrs/week",
        completion_months="Any",
        budget_preference="Any",
        delivery_preference="Any",
        certificate_preference="Any",
        preferred_platform="Any",
        preferred_language="Any",
        category="Any",
        limit=5,
        user_id="",
        interaction_revision=0,
    )
    gap["recommended_courses"] = recommendations["recommendations"]
    return gap


@app.get("/evaluate")
def evaluate(
    recommended: str = Query(default="", description="Comma-separated recommended course ids or names."),
    relevant: str = Query(default="", description="Comma-separated relevant course ids or names."),
    k: int = Query(default=5, ge=1, le=50),
) -> dict[str, float]:
    recommended_ids = [item.strip() for item in recommended.split(",") if item.strip()]
    relevant_ids = {item.strip() for item in relevant.split(",") if item.strip()}
    return {
        "precision_at_k": precision_at_k(recommended_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(recommended_ids, relevant_ids, k),
        "map_at_k": average_precision(recommended_ids, relevant_ids, k),
        "ndcg_at_k": ndcg_at_k(recommended_ids, relevant_ids, k),
    }


@app.get("/users/{user_id}/profile")
def get_user_profile(user_id: str) -> dict[str, object]:
    return PROFILE_STORE.get(user_id)


@app.post("/users/{user_id}/events")
def record_user_event(
    user_id: str,
    event_type: str = Query(default="view", pattern="^(view|click|save|enroll|complete)$"),
    course: str = Query(default="", max_length=300),
    course_id: str = Query(default="", max_length=120),
    category: str = Query(default="", max_length=120),
    sub_category: str = Query(default="", max_length=120),
    platform: str = Query(default="", max_length=120),
    level: str = Query(default="", max_length=80),
    session_id: str = Query(default="", max_length=120),
    source: str = Query(default="frontend", max_length=80),
) -> dict[str, object]:
    try:
        return PROFILE_STORE.record_event(
            user_id=user_id,
            event_type=event_type,
            course=course,
            course_id=course_id,
            category=category,
            sub_category=sub_category,
            platform=platform,
            level=level,
            session_id=session_id,
            source=source,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/users/{user_id}/interactions")
def get_user_interactions(user_id: str) -> dict[str, object]:
    history = INTERACTION_STORE.user_history(user_id)
    return {
        "user_id": user_id,
        "count": int(len(history)),
        "interactions": history.to_dict("records"),
    }


@app.get("/models/versions")
def get_model_versions() -> dict[str, object]:
    return model_metadata()


@app.post("/experiments/benchmark")
def benchmark(k: int = Query(default=10, ge=1, le=50)) -> dict[str, object]:
    courses, vectorizer, tfidf_matrix, _ = _load_assets()
    report = run_benchmark(courses, vectorizer, tfidf_matrix, k=k)
    json_path, html_path = write_reports(report, Path(__file__).resolve().parent / "reports")
    report["report_paths"] = {
        "json": str(json_path),
        "dashboard": str(html_path),
    }
    return report
