"""Turns uploaded resume bytes into clean, reading-order text/Markdown.

Production should route through a layout-aware model (Marker, Amazon Textract)
since PyPDF-style extraction scrambles multi-column resumes. The fallback
parsers here are dependency-light and good enough for single-column text and
for tests; swap `get_document_parser()` to a Marker/Textract-backed
implementation for production multi-column PDFs.
"""
from typing import Protocol


class DocumentParser(Protocol):
    def parse(self, file_bytes: bytes, filename: str) -> str: ...


class PlainTextFallbackParser:
    """Best-effort local parser: pypdf for PDF, python-docx for DOCX, else utf-8 decode."""

    def parse(self, file_bytes: bytes, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return self._parse_pdf(file_bytes)
        if lower.endswith(".docx"):
            return self._parse_docx(file_bytes)
        return file_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> str:
        from io import BytesIO

        from docx import Document

        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)


class MarkerParser:
    """Layout-aware PDF -> Markdown conversion via the `marker` package.

    Requires `pip install marker-pdf`. Not installed by default because it
    pulls in a full deep-learning stack; this class is the production swap-in.
    """

    def parse(self, file_bytes: bytes, filename: str) -> str:
        from marker.convert import convert_single_pdf  # type: ignore
        from marker.models import load_all_models  # type: ignore

        models = load_all_models()
        text, *_ = convert_single_pdf(file_bytes, models)
        return text


def get_document_parser() -> DocumentParser:
    return PlainTextFallbackParser()
