# AI Job Recommendation System

A local application that compares resume text with the job listings in `data/jobs.csv`. It uses TF-IDF word and phrase similarity plus a transparent skill coverage component. A match score is a relative ranking signal, not a probability of being hired. The current version does not assess education, seniority, or experience requirements.

## Features and architecture

- `backend/recommend.py`: CSV validation, skill extraction, TF-IDF ranking, explanations, and the original command-line entry point.
- `backend/resume_parser.py`: in-memory text extraction from PDF and DOCX, with a 5 MB cap and format checks.
- `backend/main.py`: FastAPI routes and static frontend hosting.
- `frontend/`: responsive resume upload/manual text UI, filters, and recommendation cards.
- `tests/`: standard-library `unittest` checks for data, ranking, empty input, and file validation.

```text
backend/
  main.py             FastAPI service and static page hosting
  recommend.py        Dataset pipeline, matching engine, and CLI
  resume_parser.py    In-memory PDF/DOCX parsing
data/
  jobs.csv            Supplied job listings
frontend/
  index.html          Resume form and job browsing UI
  app.js              API requests, filtering, and result rendering
  style.css           Responsive visual styles
tests/
  test_recommend.py   Engine, parsing, and API route tests
requirements.txt      Runtime dependencies
```

The score formula is `0.7 × TF-IDF cosine similarity + 0.3 × skill coverage`, with each component scaled from 0 to 100. Skill coverage is the fraction of dictionary-recognized skills in a listing that were found in the resume. The maintainable starter dictionary is in `backend/recommend.py`; add domain terms there as needed.

## Dataset

The existing `data/jobs.csv` is retained. The required field is `title`; `company`, `location`, `skills`, and `description` are supported and missing optional columns are treated as empty. Extra columns such as `job_id` and `experience_required` are retained by the loader. Duplicate title/company/location rows are removed. The current ten records are sample listings in the supplied dataset; verify their source before presenting them as live vacancies.

## Windows PowerShell setup

```powershell
cd C:\Users\yashp\ai-job-recommendation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the provided `.venv` is already active and working, use it instead of recreating it. Start the application:

```powershell
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`. API documentation is at `/docs`.

The original CLI remains available:

```powershell
python backend/recommend.py
```

## API

- `GET /`: web interface.
- `GET /health`: service status.
- `GET /jobs?title=Data&location=Remote&limit=100`: browse and filter available rows.
- `POST /recommend`: JSON resume text or multipart PDF/DOCX upload. `top_k` accepts values from 1 to 50.

JSON example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend -ContentType 'application/json' -Body '{"resume_text":"Python, SQL, pandas and machine learning","top_k":5}'
```

Upload example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/recommend -F "file=@C:\path\to\resume.pdf" -F "top_k=5"
```

The response includes ranking score, text similarity, skill coverage, matched/missing dictionary skills, and the scoring formula. Resume contents are not persisted or logged by this application.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations

This is a local portfolio project, not a production service. PDF text extraction cannot read scanned images without OCR. Skill matching is dictionary-based and may miss synonyms or infer irrelevant matches. TF-IDF does not understand semantic equivalence. There is no authentication, persistent database, or external job feed; uploaded files are processed in memory. Confirm listing freshness and legitimacy independently.
