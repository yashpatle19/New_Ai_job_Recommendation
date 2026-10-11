"""TF-IDF and skill-overlap job recommendation engine and backwards-compatible CLI."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "jobs.csv"
REQUIRED_COLUMNS = ("title", "company", "location", "skills", "description")
SCORE_FORMULA = "0.7 × TF-IDF cosine similarity + 0.3 × skill coverage; components are scaled to 0–100."
SKILL_DICTIONARY = {
    "python", "java", "javascript", "typescript", "react", "next.js", "html", "css",
    "sql", "postgresql", "mysql", "excel", "power bi", "pandas", "numpy", "scikit-learn",
    "machine learning", "deep learning", "computer vision", "opencv", "pytorch", "tensorflow",
    "nlp", "transformers", "statistics", "fastapi", "rest api", "docker", "kubernetes", "mlflow",
    "microservices", "data analysis", "predictive modeling",
}


def extract_skills(text: str, dictionary: Iterable[str] = SKILL_DICTIONARY) -> set[str]:
    """Extract dictionary terms using case-insensitive whole phrase matches."""
    normalized = (text or "").lower()
    return {skill for skill in dictionary if re.search(r"(?<![\w+#.])" + re.escape(skill.lower()) + r"(?![\w+#.])", normalized)}


def load_jobs(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate and deduplicate jobs without requiring optional columns."""
    try:
        jobs = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS])
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Could not read the job dataset.") from exc
    if jobs.empty:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS])
    if "title" not in jobs.columns:
        raise ValueError("Job dataset is missing required column: title")
    for column in REQUIRED_COLUMNS:
        if column not in jobs.columns:
            jobs[column] = ""
        jobs[column] = jobs[column].fillna("").astype(str).str.strip()
    jobs = jobs[jobs.title.ne("")].drop_duplicates(subset=["title", "company", "location"], keep="first").reset_index(drop=True)
    return jobs


def recommend_jobs(resume_text: str, top_k: int = 5, data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Rank jobs with 70% textual similarity and 30% resume skill coverage."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text must not be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    jobs = load_jobs(data_path)
    output_columns = ["title", "company", "location", "skills", "match_score", "text_similarity", "skill_coverage", "matched_skills", "missing_skills", "explanation"]
    if jobs.empty:
        return pd.DataFrame(columns=output_columns)
    jobs["search_text"] = jobs[["title", "skills", "description"]].agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        vectors = vectorizer.fit_transform(jobs.search_text.tolist() + [resume_text])
        similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    except ValueError:  # all job/resume text is empty or tokenizes to nothing
        similarities = [0.0] * len(jobs)
    resume_skills = extract_skills(resume_text)
    scores = []
    for idx, row in jobs.iterrows():
        job_skills = extract_skills(row.skills)
        matched = sorted(resume_skills & job_skills)
        missing = sorted(job_skills - resume_skills)
        coverage = len(matched) / len(job_skills) if job_skills else 0.0
        similarity = float(similarities[idx])
        score = 100 * (0.7 * similarity + 0.3 * coverage)
        scores.append({"match_score": score, "text_similarity": similarity * 100, "skill_coverage": coverage * 100,
                       "matched_skills": matched, "missing_skills": missing,
                       "explanation": f"{SCORE_FORMULA} Matched {len(matched)} of {len(job_skills)} listed skills."})
    enriched = pd.concat([jobs, pd.DataFrame(scores)], axis=1)
    return enriched.sort_values(["match_score", "title"], ascending=[False, True], kind="stable").head(top_k)[output_columns].reset_index(drop=True).round({"match_score": 2, "text_similarity": 2, "skill_coverage": 2})


if __name__ == "__main__":
    sample_resume = "Computer Science student skilled in Python, machine learning, pandas, scikit-learn, SQL, predictive modeling, and data analysis."
    recommendations = recommend_jobs(sample_resume)
    print("\nRecommended Jobs:\n")
    print(recommendations[["title", "company", "location", "skills", "match_score"]].to_string(index=False))
