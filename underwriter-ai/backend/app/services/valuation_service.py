"""Valuation orchestration: comparables and DCF on top of the extracted DRHP financials.

DRHP-derived inputs (revenue, EBITDA, PAT, net debt) are filled from the extracted values
with their source pages; anything the user supplies overrides them and is labelled
"user input". The latest full financial year is used by default (interim periods are
not annualised).
"""

from app.core.compliance import VALUATION_NOTICE
from app.financial_engine.extractor import FinancialData
from app.services.financial_extraction_service import load_financials
from app.valuation.comparables import PeerCompany, TargetFinancials, value_by_comparables
from app.valuation.dcf import DCFAssumptions, run_dcf


def latest_full_year(data: FinancialData) -> str | None:
    fys = [p for p in data.periods if p.startswith("FY")]
    return max(fys, key=lambda p: int(p[2:])) if fys else (data.periods[0] if data.periods else None)


def target_from_financials(data: FinancialData, period: str | None = None,
                           shares_outstanding: float | None = None) -> TargetFinancials:
    period = period or latest_full_year(data)
    src: dict[str, str] = {}

    def take(fld: str, key: str) -> float | None:
        v = data.get(fld, period)
        if v is None or v.value_inr is None:
            return None
        src[key] = f"DRHP p.{v.source_page}" + (" (derived: " + (v.formula or "") + ")" if v.derived else "")
        return v.value_inr

    t = TargetFinancials(period=period, revenue=take("revenue_from_operations", "revenue"), ebitda=take("ebitda", "ebitda"),
                         pat=take("profit_after_tax", "pat"), net_debt=take("net_debt", "net_debt"),
                         shares_outstanding=shares_outstanding, sources=src)
    if shares_outstanding:
        src["shares_outstanding"] = "user input"
    return t


def comparable_valuation(document_id: str, peers: list[PeerCompany], period: str | None = None,
                         shares_outstanding: float | None = None, overrides: dict | None = None) -> dict:
    data = load_financials(document_id)
    if data is None:
        raise LookupError("No extracted financials for this document.")
    target = target_from_financials(data, period, shares_outstanding)
    for k, v in (overrides or {}).items():
        if v is not None and hasattr(target, k):
            setattr(target, k, v)
            target.sources[k] = "user input"
    res = value_by_comparables(target, peers).to_dict()
    res["notice"] = VALUATION_NOTICE
    return res


def dcf_defaults(document_id: str) -> dict:
    """Base-year inputs from the DRHP, to pre-fill the DCF form (assumptions left to the user)."""
    data = load_financials(document_id)
    if data is None:
        raise LookupError("No extracted financials for this document.")
    p = latest_full_year(data)
    rev, ebitda, nd = data.get("revenue_from_operations", p), data.get("ebitda", p), data.get("net_debt", p)
    dep = data.get("depreciation", p)
    capex = data.get("capex", p)
    out = {"base_period": p, "input_sources": {}}
    if rev:
        out["base_revenue"] = rev.value_inr
        out["input_sources"]["base_revenue"] = f"DRHP p.{rev.source_page}"
        if ebitda:
            out["ebitda_margin_historical"] = ebitda.value_inr / rev.value_inr
        if dep:
            out["da_pct_revenue_historical"] = dep.value_inr / rev.value_inr
        if capex:
            out["capex_pct_revenue_historical"] = abs(capex.value_inr) / rev.value_inr
    if nd:
        out["net_debt"] = nd.value_inr
        out["input_sources"]["net_debt"] = f"DRHP p.{nd.source_page} (derived)"
    return out


def dcf_valuation(assumptions: DCFAssumptions) -> dict:
    res = run_dcf(assumptions).to_dict()
    res["notice"] = VALUATION_NOTICE
    return res
