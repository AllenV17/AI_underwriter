"""Units, statement parsing, field mapping and validation."""

import pymupdf
import pytest

from app.financial_engine.extractor import map_fields
from app.financial_engine.statement_parser import locate_and_parse, norm_label, parse_period
from app.financial_engine.units import detect_unit, parse_number, to_inr


@pytest.mark.parametrize("text,unit", [
    ("(₹ in million)", "million"), ("(Rs. in Lakhs)", "lakh"), ("All amounts are in INR millions", "million"),
    ("(All amounts are in Indian ₹ million except share data)", "million"), ("(Amount in ₹ crore)", "crore"),
    ("` in Millions", "million"), ("(₹ in thousands)", "thousand"),
])
def test_detect_unit(text, unit):
    assert detect_unit(text)[0] == unit


def test_detect_unit_absent():
    assert detect_unit("Statement of profit and loss") is None


@pytest.mark.parametrize("tok,val", [("1,234.50", 1234.5), ("(1,234.50)", -1234.5), ("5,10,912.03", 510912.03),
                                      ("-", None), ("Nil", None), ("12", 12.0)])
def test_parse_number(tok, val):
    assert parse_number(tok) == val


def test_to_inr():
    assert to_inr(2.5, "crore") == 25_000_000
    assert to_inr(1, "lakh") == 100_000


@pytest.mark.parametrize("text,period", [
    ("Year ended March 31, 2024", "FY2024"), ("31 March 2023", "FY2023"), ("December 31, 2022", "Dec 2022"),
    ("Fiscal 2025", "FY2025"), ("2023-24", "FY2024"), ("For the six months ended September 30, 2023", "Sep 2023"),
])
def test_parse_period(text, period):
    assert parse_period(text) == period


@pytest.mark.parametrize("raw,clean", [
    ("iv Profit/(loss) before tax", "profit/(loss) before tax"), ("(a) Revenue from operations", "revenue from operations"),
    ("IV EXPENSES i) Finance cost", "finance cost"), ("Total income (I)", "total income (i)"),
])
def test_norm_label(raw, clean):
    assert norm_label(raw) == clean


def test_statements_from_sample_pdf(sample_pdf):
    n = len(pymupdf.open(sample_pdf))
    tables = locate_and_parse(str(sample_pdf), list(range(6, 9)), n)
    assert set(tables) == {"pl", "bs", "cf"}
    assert tables["pl"].periods == ["FY2025", "FY2024"]
    assert tables["pl"].unit == "million"
    data = map_fields(tables)
    assert data.inr("revenue_from_operations", "FY2025") == pytest.approx(1_000e6)
    assert data.inr("profit_after_tax", "FY2024") == pytest.approx(73e6)
    assert data.inr("eps_basic", "FY2025") == pytest.approx(15.0)  # EPS stays in rupees, not scaled
    # EBITDA = PBT + finance + D&A - other income = 200 + 30 + 40 - 20
    ebitda = data.get("ebitda", "FY2025")
    assert ebitda.derived and ebitda.value_inr == pytest.approx(250e6)
    assert data.inr("net_debt", "FY2025") == pytest.approx(190e6)
    assert data.inr("free_cash_flow", "FY2025") == pytest.approx(110e6)
    assert all(c.passed for c in data.checks if c.passed is not None)
    assert data.get("revenue_from_operations", "FY2025").source_page == 6


def test_missing_values_are_not_zero(sample_pdf):
    n = len(pymupdf.open(sample_pdf))
    data = map_fields(locate_and_parse(str(sample_pdf), list(range(6, 9)), n))
    assert data.get("purchases_stock_in_trade", "FY2025") is None  # not printed -> absent, not 0
