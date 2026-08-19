"""Turns uploaded resume bytes into clean, reading-order text/Markdown.

Production should route through a layout-aware model (Marker, Amazon Textract)
since PyPDF-style extraction scrambles multi-column resumes. The fallback
parsers here are dependency-light and good enough for single-column text and
for tests; swap `get_document_parser()` to a Marker/Textract-backed
implementation for production multi-column PDFs.
"""
from typing import Iterator, Protocol


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

        document = Document(BytesIO(file_bytes))
        return "\n".join(_iter_docx_blocks(document))


def _iter_docx_blocks(container) -> Iterator[str]:
    """Walks a docx body in document order, descending into tables.

    `Document.paragraphs` skips table cells entirely, so a resume laid out in a
    table yields nothing but the name — every skill and role silently dropped
    with no error to notice.
    """
    from docx.document import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    element = container.element.body if isinstance(container, DocxDocument) else container._tc

    for child in element.iterchildren():
        if child.tag == qn("w:p"):
            text = Paragraph(child, container).text.strip()
            if text:
                yield text
        elif child.tag == qn("w:tbl"):
            for row in Table(child, container).rows:
                # Cells are joined on one line so a role and its dates stay together.
                cells = [" ".join(_iter_docx_blocks(cell)).strip() for cell in row.cells]
                line = " ".join(cell for cell in cells if cell).strip()
                if line:
                    yield line


def get_document_parser() -> DocumentParser:
    return PlainTextFallbackParser()
