"""Financial extraction: locate statements, normalise values, validate, persist."""

import logging
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.financial_engine.extractor import FinancialData, map_fields
from app.financial_engine.statement_parser import StatementTable, locate_and_parse
from app.models.db import FinancialValueRow
from app.schemas.document_ir import DocumentIR
from app.utils.storage import financials_path, load_json, save_json

log = logging.getLogger(__name__)


def candidate_pages(ir: DocumentIR) -> list[int]:
    """Pages of the restated-financials chapter(s), longest chapter first.

    Some DRHPs have a short summary chapter that also matches "financial information";
    the full restated statements sit in the longest such chapter.
    """
    chs = sorted([c for c in ir.chapters if c.key == "restated_financials"], key=lambda c: c.pdf_start - c.pdf_end)
    pages: list[int] = []
    for c in chs:
        pages.extend(range(c.pdf_start, c.pdf_end + 1))
    return pages


def extract_financials(ir: DocumentIR, pdf_path: Path) -> tuple[FinancialData, dict[str, StatementTable]]:
    tables = locate_and_parse(str(pdf_path), candidate_pages(ir), ir.n_pages)
    data = map_fields(tables) if tables else FinancialData(periods=[], basis="unknown", unit=None,
                                                          warnings=["No primary financial statements were located."])
    missing = [s for s in ("pl", "bs", "cf") if s not in tables]
    if missing:
        data.warnings.append(f"Statements not located: {', '.join(missing)}")
    return data, tables


def persist_financials(session: Session, document_id: str, data: FinancialData) -> None:
    save_json(financials_path(document_id), data.to_dict())
    session.execute(delete(FinancialValueRow).where(FinancialValueRow.document_id == document_id))
    for v in data.values:
        session.add(FinancialValueRow(document_id=document_id, **{k: getattr(v, k) for k in (
            "field", "label", "period", "value_original", "unit", "value_inr", "statement", "basis",
            "source_page", "source_label", "derived", "formula")}))
    session.commit()


def load_financials(document_id: str) -> FinancialData | None:
    """Rebuild FinancialData from the saved JSON."""
    from app.financial_engine.extractor import FinancialValue, ValidationCheck

    raw = load_json(financials_path(document_id))
    if raw is None:
        return None
    return FinancialData(
        periods=raw["periods"], basis=raw["basis"], unit=raw["unit"],
        values=[FinancialValue(**v) for v in raw["values"]], checks=[ValidationCheck(**c) for c in raw["checks"]],
        warnings=raw["warnings"], statement_pages=raw.get("statement_pages", {}),
    )
