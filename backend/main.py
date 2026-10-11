"""FastAPI application for job browsing and resume recommendations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from backend.recommend import DATA_PATH, SCORE_FORMULA, load_jobs, recommend_jobs
from backend.resume_parser import MAX_UPLOAD_BYTES, extract_resume_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="AI Job Recommendation System", version="1.0.0", description="Explainable resume-to-job matching using TF-IDF and skill overlap.")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "frontend"), name="static")


class RecommendRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=100_000)
    top_k: int = Field(default=5, ge=1, le=50)


@app.get("/")
def root() -> Any:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs")
def get_jobs(location: str | None = None, title: str | None = None, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit must be between 1 and 500")
    try:
        jobs = load_jobs(DATA_PATH)
    except ValueError as exc:
        raise HTTPException(500, "Job dataset could not be loaded.") from exc
    if location:
        jobs = jobs[jobs.location.str.contains(location, case=False, regex=False)]
    if title:
        jobs = jobs[jobs.title.str.contains(title, case=False, regex=False)]
    return {"jobs": jobs.head(limit).to_dict(orient="records"), "count": min(len(jobs), limit)}


@app.post("/recommend")
async def post_recommend(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            try:
                payload = RecommendRequest.model_validate(await request.json())
            except (ValidationError, ValueError) as exc:
                raise HTTPException(422, "Provide non-empty resume_text and top_k from 1 to 50.") from exc
            resume_text, top_k = payload.resume_text, payload.top_k
        elif "multipart/form-data" in content_type:
            if int(request.headers.get("content-length", "0") or 0) > MAX_UPLOAD_BYTES + 100_000:
                raise HTTPException(413, "Upload exceeds the 5 MB limit.")
            form = await request.form()
            try:
                top_k = int(form.get("top_k", 5))
                if not 1 <= top_k <= 50:
                    raise HTTPException(422, "top_k must be between 1 and 50.")
                upload = form.get("file")
                manual_text = form.get("resume_text")
                if upload is not None and hasattr(upload, "read"):
                    content = await upload.read(MAX_UPLOAD_BYTES + 1)
                    resume_text = extract_resume_text(upload.filename or "", content)
                elif isinstance(manual_text, str):
                    resume_text = manual_text
                else:
                    raise HTTPException(422, "Upload a PDF/DOCX file or provide resume_text.")
            finally:
                await form.close()
        else:
            raise HTTPException(415, "Use application/json or multipart/form-data.")
        results = recommend_jobs(resume_text, top_k)
        return {"recommendations": results.to_dict(orient="records"), "count": len(results), "scoring_formula": SCORE_FORMULA}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        # Keep details and user supplied resume contents out of responses/logs.
        raise HTTPException(500, "Recommendation request could not be processed.") from exc
