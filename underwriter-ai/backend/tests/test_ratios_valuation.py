"""Ratio engine, comparables and DCF."""

import pymupdf
import pytest

from app.financial_engine.extractor import FinancialData, FinancialValue, map_fields
from app.financial_engine.ratios import RatioEngine, is_financial_company
from app.financial_engine.statement_parser import locate_and_parse
from app.valuation.comparables import PeerCompany, TargetFinancials, value_by_comparables
from app.valuation.dcf import DCFAssumptions, run_dcf


@pytest.fixture(scope="module")
def data(sample_pdf) -> FinancialData:
    n = len(pymupdf.open(sample_pdf))
    return map_fields(locate_and_parse(str(sample_pdf), list(range(6, 9)), n))


def _by(ratios, name, period="FY2025"):
    return next(r for r in ratios if r.name == name and r.period == period)


def test_ratio_values(data):
    rs = RatioEngine(data).compute()
    assert _by(rs, "Net Profit Margin").value == pytest.approx(15.0)
    assert _by(rs, "Return on Equity").value == pytest.approx(25.0)
    assert _by(rs, "Current Ratio").value == pytest.approx(2.0)
    assert _by(rs, "Quick Ratio").value == pytest.approx(1.4)
    assert _by(rs, "Debt-to-Equity").value == pytest.approx(250 / 600, abs=1e-4)
    assert _by(rs, "Interest Coverage Ratio").value == pytest.approx(230 / 30, abs=1e-3)
    assert _by(rs, "EBITDA Margin").value == pytest.approx(25.0)
    r = _by(rs, "Return on Equity")
    assert r.formula and r.inputs[0].source_page == 6 and r.interpretation


def test_ratio_missing_input_reason(data):
    rs = RatioEngine(data).compute()
    pe = _by(rs, "P/E")
    assert not pe.available and "price per share" in pe.reason.lower()


def test_ratio_with_price(data):
    rs = RatioEngine(data, price_per_share=300.0, shares_outstanding=10e6).compute()
    assert _by(rs, "P/E").value == pytest.approx(20.0)
    assert _by(rs, "Price-to-Book").value == pytest.approx(3000e6 / 600e6)


def test_financial_company_detection():
    d = FinancialData(periods=["FY2024"], basis="standalone", unit="million")
    for f, v in (("revenue_from_operations", 100e6), ("finance_costs", 60e6)):
        d.values.append(FinancialValue(f, f, "FY2024", v / 1e6, "million", v, "pl", "standalone", 1, f))
    assert is_financial_company(d)
    rs = RatioEngine(d).compute()
    assert not _by(rs, "EBITDA Margin", "FY2024").available


def test_comparables():
    peers = [PeerCompany(name="A", market_cap=2000e6, pat=100e6, revenue=1000e6),
             PeerCompany(name="B", pe=25, ps=3),
             PeerCompany(name="C", pe=-5),  # loss-maker excluded
             PeerCompany(name="D", pe=15, ev_ebitda=12)]
    t = TargetFinancials(period="FY2025", revenue=1000e6, ebitda=250e6, pat=150e6, net_debt=190e6, shares_outstanding=10e6)
    res = value_by_comparables(t, peers)
    pe = next(r for r in res.results if r.multiple == "P/E")
    assert pe.median == pytest.approx(20.0) and "C" in pe.excluded
    assert pe.implied_price_mid == pytest.approx(300.0)
    ev = next(r for r in res.results if r.multiple == "EV/EBITDA")
    assert ev.implied_equity_mid == pytest.approx(12 * 250e6 - 190e6)
    assert res.price_range[0] <= res.price_range[1] <= res.price_range[2]


def test_comparables_needs_peers():
    res = value_by_comparables(TargetFinancials(period="FY2025", pat=1.0), [])
    assert res.equity_range is None


def test_dcf_matches_hand_calculation():
    a = DCFAssumptions(base_revenue=1000.0, revenue_growth=[0.10], ebitda_margin=0.20, tax_rate=0.25, da_pct_revenue=0.05,
                       capex_pct_revenue=0.05, nwc_pct_incremental_revenue=0.10, wacc=0.12, terminal_growth=0.04,
                       net_debt=100.0, shares_outstanding=10.0)
    r = run_dcf(a)
    # rev 1100, ebitda 220, da 55, ebit 165, nopat 123.75, capex 55, dnwc 10 -> fcff 113.75
    assert r.years[0].fcff == pytest.approx(113.75)
    tv = 113.75 * 1.04 / 0.08
    assert r.enterprise_value == pytest.approx(113.75 / 1.12 + tv / 1.12)
    assert r.price_per_share == pytest.approx((r.enterprise_value - 100) / 10)
    assert len(r.sensitivity_wacc_g["values"]) == 5


def test_dcf_rejects_growth_above_wacc():
    with pytest.raises(ValueError):
        DCFAssumptions(base_revenue=1, revenue_growth=[0.1], ebitda_margin=0.2, wacc=0.08, terminal_growth=0.09)


def test_comparables_pre_money_vs_post_money():
    peers = [PeerCompany(name=n, pe=20) for n in "ABC"]
    base = dict(period="FY2025", pat=1000e6)  # pre-money equity = 20 x 1000m = 20,000m
    # Pre-issue 100m shares -> Rs. 200 per share, regardless of the new shares.
    r = value_by_comparables(TargetFinancials(**base, pre_issue_shares=100e6), peers)
    assert r.price_range[1] == pytest.approx(200.0)
    # Post-issue 120m shares with Rs. 4,000m fresh issue: (20,000 + 4,000) / 120 = 200, the same price.
    r = value_by_comparables(TargetFinancials(**base, shares_outstanding=120e6, fresh_issue_inr=4000e6), peers)
    assert r.price_range[1] == pytest.approx(200.0) and not r.warnings
    # Post-issue count alone understates the price (20,000 / 120 = 166.7) and says so.
    r = value_by_comparables(TargetFinancials(**base, shares_outstanding=120e6), peers)
    assert r.price_range[1] == pytest.approx(20000 / 120) and any("pre-issue" in w for w in r.warnings)


def test_dcf_adds_fresh_issue_proceeds():
    kw = dict(base_revenue=1000.0, revenue_growth=[0.10], ebitda_margin=0.20, tax_rate=0.25, da_pct_revenue=0.05,
              capex_pct_revenue=0.05, wacc=0.12, terminal_growth=0.04, net_debt=100.0, shares_outstanding=10.0)
    without, withfresh = run_dcf(DCFAssumptions(**kw)), run_dcf(DCFAssumptions(**kw, fresh_issue_inr=50.0))
    assert withfresh.equity_value == pytest.approx(without.equity_value + 50.0)
    assert any("Fresh-issue" in w for w in without.warnings) and not any("Fresh-issue" in w for w in withfresh.warnings)
