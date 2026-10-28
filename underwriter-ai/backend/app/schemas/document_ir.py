"""Intermediate representation (IR) of a processed DRHP.

The IR is the single hand-off format between ingestion and every consumer: the RAG
chunker reads `pages`, the financial engine reads `tables` and the section map. It is
stored as JSON under data/processed/<document_id>/document.json.
"""

from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """One chapter from the document's table of contents, mapped to PDF pages (1-based)."""

    section: str | None = None
    title: str
    key: str
    printed_start: int
    pdf_start: int
    pdf_end: int
    title_verified: bool = False


class PageIR(BaseModel):
    page_number: int = Field(description="1-based PDF page number")
    printed_page: int | None = None
    section: str | None = None
    chapter: str | None = None
    chapter_key: str = "front_matter"
    subsection: str | None = None
    text: str
    ocr: bool = False


class TableIR(BaseModel):
    """A table extracted from a page, kept whole so row/column relationships survive."""

    table_id: str
    page_number: int
    page_numbers: list[int] = Field(default_factory=list, description="All pages the table spans")
    chapter_key: str
    title: str | None = None
    unit_text: str | None = None
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    engine: str = "pdfplumber"

    def to_markdown(self, max_rows: int = 80) -> str:
        cols = max([len(self.header)] + [len(r) for r in self.rows] + [1])
        head = self.header + [""] * (cols - len(self.header)) if self.header else [f"col{i + 1}" for i in range(cols)]
        lines = ["| " + " | ".join(h.replace("\n", " ") for h in head) + " |", "|" + " --- |" * cols]
        for r in self.rows[:max_rows]:
            r = r + [""] * (cols - len(r))
            lines.append("| " + " | ".join((c or "").replace("\n", " ") for c in r) + " |")
        return "\n".join(lines)


class DocumentIR(BaseModel):
    document_id: str
    company_name: str | None = None
    filename: str
    n_pages: int
    doc_type: str | None = Field(default=None, description="DRHP / RHP / Prospectus, from the cover page")
    page_offset: int | None = None
    page_offset_agreement: float = 0.0
    chapters: list[Chapter] = Field(default_factory=list)
    pages: list[PageIR] = Field(default_factory=list)
    tables: list[TableIR] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def page(self, number: int) -> PageIR | None:
        return self.pages[number - 1] if 1 <= number <= len(self.pages) else None

    def pages_in(self, *keys: str) -> list[PageIR]:
        return [p for p in self.pages if p.chapter_key in keys]
