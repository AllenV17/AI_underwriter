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
    # IREDA's cash flow prints subtotals indented from the line items and labels OCF "Net Cash Generated from Operations".
    ("IREDA_DRHP.pdf", "FY2023", "operating_cash_flow", -123430.78, "million", 359),
    ("IREDA_DRHP.pdf", "Jun 2023", "operating_cash_flow", 4097.68, "million", 359),
    # Honasa's page title sits above a multi-line period header; EPS is labelled by face value only.
    ("HonasaConsumer_DRHP.pdf", "FY2021", "revenue_from_operations", 4599.90, "million", 73),
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


def test_honasa_periods_and_eps():
    data = _load("HonasaConsumer_DRHP.pdf")
    assert data.periods == ["Sep 2022", "FY2022", "FY2021", "FY2020"]
    assert data.get("eps_basic", "FY2022").value_original == pytest.approx(0.53)


def _peers(name: str):
    import pymupdf

    from app.financial_engine.peer_extractor import extract_peers

    path = PILOT / name
    if not path.exists():
        pytest.skip(f"pilot DRHP {name} not available")
    texts = [p.get_text() for p in pymupdf.open(path)]
    chapters, *_ = map_sections(texts)
    pages = [n for c in chapters if c.key == "basis_for_offer_price" for n in range(c.pdf_start, c.pdf_end + 1)]
    return extract_peers(str(path), pages)


def test_peer_pe_mankind():
    pes = sorted(p.values["pe"] for p in _peers("MankindPharma_DRHP.pdf").peers if not p.is_subject)
    assert {24.99, 48.46, 14.27, 58.48, 37.01, 34.05} <= set(pes)


def test_peer_pe_doms_and_none_cases():
    assert 25.9 in [p.values["pe"] for p in _peers("DOMSIndustries_DRHP.pdf").peers]
    t = _peers("HonasaConsumer_DRHP.pdf")
    assert not t.peers and "no comparable" in t.note


def test_peer_filters_and_notes():
    # A real peer whose name contains generic words must not be mistaken for the issuer (Aptus Value Housing).
    t = _peers("BajajHousingFinance_DRHP.pdf")
    names = [p.name for p in t.peers if not p.is_subject]
    assert any("Aptus Value" in n for n in names)
    # The note about missing peers is only attached when no table was parsed.
    assert t.peers and "no comparable" not in t.note
