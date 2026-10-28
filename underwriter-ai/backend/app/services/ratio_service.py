"""Ratio calculation for a processed document."""

from app.financial_engine.ratios import RatioEngine, RatioResult, is_financial_company
from app.services.financial_extraction_service import load_financials


class FinancialsNotAvailableError(LookupError):
    pass


def compute_ratios(document_id: str, price_per_share: float | None = None,
                   shares_outstanding: float | None = None) -> tuple[list[RatioResult], dict]:
    data = load_financials(document_id)
    if data is None or not data.values:
        raise FinancialsNotAvailableError("No extracted financial statements for this document.")
    engine = RatioEngine(data, price_per_share, shares_outstanding)
    meta = {"periods": data.periods, "basis": data.basis, "unit": data.unit,
            "financial_company": is_financial_company(data),
            "checks": [c.__dict__ for c in data.checks], "warnings": data.warnings}
    return engine.compute(), meta
