"""Integration tests on the real pilot DRHPs (skipped when the PDFs are not present).

Expected values were read from the cited statement pages of each filing.
"""

from pathlib import Path

import pytest

from app.financial_engine.extractor import map_fields
from app.financial_engine.statement_parser import locate_and_parse
from app.ingestion.section_mapper import map_sections

PILOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "drhp"

# (file, period, field, value as printed, unit, statement page)
CASES = [
    ("BajajHousingFinance_DRHP.pdf", "FY2024", "profit_after_tax", 17312.2, "million", 283),
    ("BajajHousingFinance_DRHP.pdf", "FY2024", "total_assets", 818270.9, "million", 282),
    ("HyundaiMotorIndia_DRHP.pdf", "Dec 2023", "revenue_from_operations", 521579.11, "million", 238),
    ("MankindPharma_DRHP.pdf", "FY2022", "profit_before_tax", 19745.99, "million", 310),
    ("IREDA_DRHP.pdf", "FY2023", "total_equity", 59351.69, "million", 357),
]


def _load(name: str):
    import pymupdf

    path = PILOT / name
    if not path.exists():
        pytest.skip(f"pilot DRHP {name} not available")
    texts = [p.get_text() for p in pymupdf.open(path)]
    chapters, *_ = map_sections(texts)
    cands = sorted([c for c in chapters if c.key == "restated_financials"], key=lambda c: c.pdf_start - c.pdf_end)
    pages = [n for c in cands for n in range(c.pdf_start, c.pdf_end + 1)]
    return map_fields(locate_and_parse(str(path), pages, len(texts)))


@pytest.mark.parametrize("name,period,field,value,unit,page", CASES)
def test_pilot_value(name, period, field, value, unit, page):
    data = _load(name)
    v = data.get(field, period)
    assert v is not None, f"{field} {period} not extracted"
    assert v.value_original == pytest.approx(value) and v.unit == unit and v.source_page == page


@pytest.mark.parametrize("name", ["BajajHousingFinance_DRHP.pdf", "HyundaiMotorIndia_DRHP.pdf", "DOMSIndustries_DRHP.pdf"])
def test_pilot_checks_pass(name):
    data = _load(name)
    assert data.checks and all(c.passed for c in data.checks if c.passed is not None)
