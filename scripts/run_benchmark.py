from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from api import _load_assets
from recommender.experiments import run_benchmark, write_reports


def main() -> None:
    courses, vectorizer, tfidf_matrix, _ = _load_assets()
    report = run_benchmark(courses, vectorizer, tfidf_matrix)
    json_path, html_path = write_reports(report, ROOT_DIR / "reports")
    print(f"JSON report: {json_path}")
    print(f"HTML dashboard: {html_path}")


if __name__ == "__main__":
    main()
