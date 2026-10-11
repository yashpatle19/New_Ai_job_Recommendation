"""Safe, in-memory extraction for text, PDF, and DOCX resumes."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx"}


def extract_resume_text(filename: str, content: bytes) -> str:
    """Validate a supported document and return its extracted text."""
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Resume file is empty or exceeds the 5 MB upload limit.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Unsupported resume format. Upload a PDF or DOCX file.")
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("The uploaded file is not a valid PDF.")
        try:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        except Exception as exc:
            raise ValueError("Could not read the PDF resume.") from exc
    else:
        if not content.startswith(b"PK"):
            raise ValueError("The uploaded file is not a valid DOCX document.")
        try:
            from docx import Document
            text = "\n".join(p.text for p in Document(BytesIO(content)).paragraphs)
        except Exception as exc:
            raise ValueError("Could not read the DOCX resume.") from exc
    if not text.strip():
        raise ValueError("No readable text was found in the resume.")
    return text
