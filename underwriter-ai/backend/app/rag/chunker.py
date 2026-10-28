"""Section-aware, page-aware chunking of the document IR.

Rules:
* A chunk never crosses a page or a chapter boundary, so every chunk cites one page.
* Paragraph chunks are built from whole paragraphs up to ~TARGET_WORDS words.
* Tables become their own chunks with title, unit and header row kept together, and are
  never split mid-row; pages whose content is a parsed financial statement are indexed
  as the table only (their raw text is a jumble of numbers).
* Each chunk carries document, company, page, section, chapter, subsection and type.
"""

import re

from pydantic import BaseModel, Field

from app.ingestion.section_mapper import KEY_LABELS
from app.schemas.document_ir import DocumentIR, TableIR

TARGET_WORDS = 280
MAX_WORDS = 420
TABLE_ROWS_PER_CHUNK = 45


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    company_name: str | None = None
    page_number: int
    section: str | None = None
    chapter: str | None = None
    chapter_key: str = "front_matter"
    subsection: str | None = None
    chunk_type: str = Field(default="paragraph", pattern="^(paragraph|table)$")
    text: str
    table_data: dict | None = None

    @property
    def section_label(self) -> str:
        return KEY_LABELS.get(self.chapter_key) or (self.chapter or "Front matter").title()

    def embedding_text(self) -> str:
        """Text sent to the embedder: a short context header plus the chunk body."""
        ctx = " | ".join(x for x in (self.company_name, self.section_label, self.subsection) if x)
        return f"{ctx}\n{self.text}" if ctx else self.text


_BOILERPLATE = re.compile(r"^\s*(\d{1,4}|page \d+|[a-z ]+limited)\s*$", re.I)


def _paragraphs(text: str) -> list[str]:
    paras = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not _BOILERPLATE.match(ln)]
        if lines:
            paras.append(re.sub(r"[ \t]+", " ", " ".join(lines)))
    return paras


def _split_long(p: str) -> list[str]:
    words = p.split()
    if len(words) <= MAX_WORDS:
        return [p]
    sentences = re.split(r"(?<=[.;:])\s+", p)
    out, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > TARGET_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    # Sentences longer than MAX_WORDS (tables flattened into text) are hard-wrapped.
    final = []
    for c in out:
        w = c.split()
        final.extend(" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS))
    return final


def chunk_document(ir: DocumentIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    statement_pages = {n for t in ir.tables if t.engine == "statement-parser" for n in (t.page_numbers or [t.page_number])}

    for page in ir.pages:
        if page.page_number in statement_pages:
            continue  # indexed as the parsed statement table instead
        base = dict(document_id=ir.document_id, company_name=ir.company_name, page_number=page.page_number,
                    section=page.section, chapter=page.chapter, chapter_key=page.chapter_key, subsection=page.subsection)
        buf: list[str] = []
        n = 0

        def flush() -> None:
            nonlocal buf, n
            if buf:
                text = "\n\n".join(buf).strip()
                if len(text.split()) >= 8:
                    chunks.append(Chunk(chunk_id=f"{ir.document_id}:p{page.page_number}:c{n}", text=text, **base))
                    n += 1
            buf = []

        for para in _paragraphs(page.text):
            for piece in _split_long(para):
                if buf and len(" ".join(buf + [piece]).split()) > TARGET_WORDS:
                    flush()
                buf.append(piece)
        flush()

    for t in ir.tables:
        chunks.extend(_table_chunks(ir, t))
    return chunks


def _table_chunks(ir: DocumentIR, t: TableIR) -> list[Chunk]:
    page = ir.page(t.page_number)
    out = []
    for i in range(0, max(len(t.rows), 1), TABLE_ROWS_PER_CHUNK):
        part = TableIR(**{**t.model_dump(), "rows": t.rows[i:i + TABLE_ROWS_PER_CHUNK]})
        head = " ".join(x for x in (t.title, f"({t.unit_text})" if t.unit_text else None) if x)
        text = f"{head}\n{part.to_markdown()}" if head else part.to_markdown()
        out.append(Chunk(
            chunk_id=f"{ir.document_id}:{t.table_id}:r{i}", document_id=ir.document_id, company_name=ir.company_name,
            page_number=t.page_number, section=page.section if page else None, chapter=page.chapter if page else None,
            chapter_key=t.chapter_key, subsection=t.title, chunk_type="table", text=text,
            table_data={"header": t.header, "rows": part.rows, "unit": t.unit_text, "table_id": t.table_id},
        ))
    return out
