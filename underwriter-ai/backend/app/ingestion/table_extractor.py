"""Table extraction for the pages where tables matter.

Running a layout model over a 500-page DRHP takes many minutes on CPU, so tables are
extracted only from the chapters that need them:

* the primary financial statements: rebuilt by the statement parser (word positions);
* "Basis for Offer Price" (peer comparison, KPIs) and the financial summary: extracted
  with pdfplumber (default) or Docling (TABLE_ENGINE=docling).

Each table is kept whole (header + rows) so the RAG chunker never splits a row from
its column headings.
"""

import logging
from pathlib import Path

from app.financial_engine.statement_parser import StatementTable
from app.financial_engine.line_items import STATEMENT_NAMES
from app.schemas.document_ir import PageIR, TableIR

log = logging.getLogger(__name__)

TABLE_CHAPTERS = ("basis_for_offer_price", "financial_summary", "capital_structure", "objects_of_offer")
MAX_TABLE_PAGES = 60


def statement_tables(tables: dict[str, StatementTable]) -> list[TableIR]:
    """Convert parsed primary statements into IR tables (one per statement page set)."""
    out = []
    for st, t in tables.items():
        out.append(TableIR(
            table_id=f"stmt-{st}", page_number=t.pages[0], page_numbers=list(t.pages), chapter_key="restated_financials",
            title=f"Restated {t.basis if t.basis != 'unknown' else ''} {STATEMENT_NAMES[st].lower()}".replace("  ", " ").strip().capitalize(),
            unit_text=t.unit_caption, header=["Particulars", *t.periods],
            rows=[[r.label, *r.raw] for r in t.rows if r.label], engine="statement-parser",
        ))
    return out


def extract_chapter_tables(pdf_path: Path, pages: list[PageIR], engine: str = "pdfplumber") -> list[TableIR]:
    """Extract tables from the pages of the chapters listed in TABLE_CHAPTERS."""
    targets = [p for p in pages if p.chapter_key in TABLE_CHAPTERS][:MAX_TABLE_PAGES]
    if not targets:
        return []
    if engine == "docling":
        try:
            return _docling_tables(pdf_path, targets)
        except Exception as e:  # fall back rather than fail ingestion
            log.warning("Docling table extraction failed (%s); falling back to pdfplumber", e)
    return _pdfplumber_tables(pdf_path, targets)


def _clean(cell: str | None) -> str:
    return " ".join((cell or "").split())


def _pdfplumber_tables(pdf_path: Path, targets: list[PageIR]) -> list[TableIR]:
    import pdfplumber

    out: list[TableIR] = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in targets:
            try:
                page = pdf.pages[p.page_number - 1]
                found = page.extract_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                if not found:
                    found = page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
            except Exception as e:
                log.debug("pdfplumber failed on page %s: %s", p.page_number, e)
                continue
            for i, tbl in enumerate(found):
                rows = [[_clean(c) for c in r] for r in tbl if r and any(_clean(c) for c in r)]
                # Discard layout artefacts: a real table has at least 2 rows and 2 non-empty columns.
                if len(rows) < 2 or max(sum(1 for c in r if c) for r in rows) < 2:
                    continue
                out.append(TableIR(table_id=f"p{p.page_number}-t{i + 1}", page_number=p.page_number,
                                   chapter_key=p.chapter_key, title=p.subsection or p.chapter,
                                   header=rows[0], rows=rows[1:], engine="pdfplumber"))
    return out


def _docling_tables(pdf_path: Path, targets: list[PageIR]) -> list[TableIR]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    out: list[TableIR] = []
    by_number = {p.page_number: p for p in targets}
    # Convert contiguous runs of pages to keep model start-up overhead down.
    numbers = sorted(by_number)
    runs: list[tuple[int, int]] = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    for a, b in runs:
        res = conv.convert(str(pdf_path), page_range=(a, b))
        for i, t in enumerate(res.document.tables):
            df = t.export_to_dataframe()
            page_no = t.prov[0].page_no if t.prov else a
            meta = by_number.get(page_no, by_number[a])
            out.append(TableIR(table_id=f"p{page_no}-d{i + 1}", page_number=page_no, chapter_key=meta.chapter_key,
                               title=meta.subsection or meta.chapter, header=[str(c) for c in df.columns],
                               rows=[[_clean(str(v)) for v in row] for row in df.astype(str).values.tolist()],
                               engine="docling"))
    return out
