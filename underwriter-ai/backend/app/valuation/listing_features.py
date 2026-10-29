"""Pre-listing features for the listing-return model, built from a document's extracted data.

One implementation serves the training table (`src/build_ipo_features.py`), the agent tool
and the report, so the model sees the same definitions at training and prediction time.
Only pre-listing information is used. Features that need manual reading of the offer
chapters or market data (issue size, fresh-issue share, Nifty return, issue P/E) are not
derived here; they stay missing and the model's imputer handles them.
"""

from statistics import median

from app.financial_engine.extractor import FinancialData
from app.financial_engine.ratios import RatioEngine


def median_peer_pe(peers: list[dict]) -> float | None:
    """Median of the peers' P/E multiples (the issuer's own row must already be excluded)."""
    pes = [p["pe"] for p in peers if p.get("pe")]
    return float(median(pes)) if pes else None


def features_from_financials(data: FinancialData) -> dict:
    """Latest full financial year: profit margin, ROE, debt-to-equity and revenue growth."""
    fys = sorted((p for p in data.periods if p.startswith("FY")), key=lambda p: int(p[2:]), reverse=True)
    if not fys:
        return {}
    res = {r.name: r.value for r in RatioEngine(data).compute() if r.period == fys[0] and r.available}
    out = {"pat_margin": res.get("Net Profit Margin"), "roe": res.get("Return on Equity"),
           "debt_to_equity": res.get("Debt-to-Equity"), "feature_period": fys[0]}
    if len(fys) > 1:
        a, b = data.inr("revenue_from_operations", fys[0]), data.inr("revenue_from_operations", fys[1])
        out["revenue_growth"] = (a / b - 1) * 100 if a and b else None
    return out
