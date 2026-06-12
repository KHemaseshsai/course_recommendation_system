from __future__ import annotations

import re


ROLE_SKILLS = {
    "Machine Learning Engineer": {
        "python",
        "statistics",
        "machine learning",
        "deep learning",
        "pandas",
        "numpy",
        "scikit-learn",
        "mlops",
        "deployment",
    },
    "Data Scientist": {"python", "sql", "statistics", "machine learning", "pandas", "visualization", "experimentation"},
    "Full Stack Developer": {"html", "css", "javascript", "react", "node", "api", "sql", "deployment"},
    "Backend Engineer": {"python", "java", "api", "sql", "database", "system design", "cloud", "deployment"},
    "Data Analyst": {"sql", "excel", "python", "statistics", "dashboard", "visualization", "data cleaning"},
}


def extract_skills(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9+#. -]", " ", text.lower())
    skills = set()
    all_skills = set().union(*ROLE_SKILLS.values())
    for skill in all_skills:
        if skill in normalized:
            skills.add(skill)
    return skills


def analyze_skill_gap(text: str, target_role: str) -> dict[str, object]:
    required = ROLE_SKILLS.get(target_role, ROLE_SKILLS["Machine Learning Engineer"])
    extracted = extract_skills(text)
    missing = sorted(required - extracted)
    return {
        "target_role": target_role,
        "extracted_skills": sorted(extracted),
        "required_skills": sorted(required),
        "missing_skills": missing,
        "recommendation_goal": " ".join(missing[:4]) if missing else target_role,
    }

