from __future__ import annotations


ROLE_PATHS = {
    "machine learning engineer": [
        "Python programming",
        "Statistics and probability",
        "Machine learning",
        "Deep learning",
        "MLOps",
        "Model deployment",
    ],
    "data scientist": [
        "Python programming",
        "SQL",
        "Statistics",
        "Data analysis",
        "Machine learning",
        "Visualization",
    ],
    "full stack developer": [
        "HTML and CSS",
        "JavaScript",
        "Frontend framework",
        "Backend APIs",
        "Databases",
        "Deployment",
    ],
    "backend engineer": [
        "Programming fundamentals",
        "Databases",
        "REST APIs",
        "System design basics",
        "Cloud deployment",
        "Monitoring",
    ],
    "data analyst": [
        "Excel or spreadsheets",
        "SQL",
        "Python basics",
        "Data cleaning",
        "Dashboards",
        "Business analytics",
    ],
}


def generate_learning_path(goal: str) -> list[str]:
    normalized_goal = goal.lower()
    for role, path in ROLE_PATHS.items():
        if role in normalized_goal:
            return path
    if "ml" in normalized_goal or "machine learning" in normalized_goal:
        return ROLE_PATHS["machine learning engineer"]
    if "data" in normalized_goal:
        return ROLE_PATHS["data scientist"]
    if "backend" in normalized_goal:
        return ROLE_PATHS["backend engineer"]
    if "full stack" in normalized_goal or "web" in normalized_goal:
        return ROLE_PATHS["full stack developer"]
    return ["Foundations", "Core skills", "Projects", "Advanced topics", "Deployment"]

