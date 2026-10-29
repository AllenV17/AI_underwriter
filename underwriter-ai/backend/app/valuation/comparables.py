"""Comparable-company (trading multiples) valuation.

Peer data is supplied by the user (or copied from the DRHP's "Basis for Offer Price" peer
table with its page cited); this module never fetches or invents peer figures. Multiples
are computed from each peer's inputs where the peer did not state them, summarised by
median (robust to outliers) with min/max as the range, and applied to the target's
extracted financials.

    P/E        -> implied equity value = median P/E x target PAT
    EV/EBITDA  -> implied EV = median EV/EBITDA x target EBITDA; equity = EV - net debt
    P/S        -> implied equity value = median P/S x target revenue
"""

from dataclasses import asdict, dataclass, field
from statistics import median

from pydantic import BaseModel, Field, model_validator


class PeerCompany(BaseModel):
    """One peer. All monetary values in absolute INR; multiples as plain numbers."""

    name: str = Field(min_length=1)
    industry: str | None = None
    revenue: float | None = None
    ebitda: float | None = None
    pat: float | None = None
    eps: float | None = None
    market_cap: float | None = Field(default=None, ge=0)
    enterprise_value: float | None = None
    pe: float | None = None
    ev_ebitda: float | None = None
    ps: float | None = None
    source: str = Field(default="user input", description="Where the figures came from, e.g. 'DRHP p.112' or 'NSE, 2024-08-30'")

    @model_validator(mode="after")
    def _fill_multiples(self) -> "PeerCompany":
        if self.pe is None and self.market_cap and self.pat and self.pat > 0:
            self.pe = self.market_cap / self.pat
        if self.ev_ebitda is None and self.enterprise_value and self.ebitda and self.ebitda > 0:
            self.ev_ebitda = self.enterprise_value / self.ebitda
        if self.ps is None and self.market_cap and self.revenue and self.revenue > 0:
            self.ps = self.market_cap / self.revenue
        return self


class TargetFinancials(BaseModel):
    """Target inputs (absolute INR). Normally taken from the extracted DRHP values."""

    period: str
    revenue: float | None = None
    ebitda: float | None = None
    pat: float | None = None
    net_debt: float | None = None
    shares_outstanding: float | None = Field(default=None, gt=0, description="Post-issue share count")
    pre_issue_shares: float | None = Field(default=None, gt=0, description="Share count before the issue (alternative to post-issue count + fresh issue)")
    fresh_issue_inr: float | None = Field(default=None, ge=0, description="Fresh-issue proceeds, absolute INR (0 for a pure offer for sale)")
    sources: dict[str, str] = Field(default_factory=dict)


@dataclass
class MultipleResult:
    multiple: str
    peer_values: dict[str, float]
    excluded: dict[str, str]
    low: float | None
    median: float | None
    high: float | None
    target_metric: str
    target_value: float | None
    implied_equity_low: float | None = None
    implied_equity_mid: float | None = None
    implied_equity_high: float | None = None
    implied_price_low: float | None = None
    implied_price_mid: float | None = None
    implied_price_high: float | None = None
    note: str = ""


@dataclass
class ComparableValuation:
    target: dict
    results: list[MultipleResult] = field(default_factory=list)
    equity_range: tuple[float, float, float] | None = None  # low, mid, high
    price_range: tuple[float, float, float] | None = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# Multiples outside these bounds are excluded as not meaningful (loss-makers, data errors).
_BOUNDS = {"P/E": (0, 200), "EV/EBITDA": (0, 100), "P/S": (0, 50)}


def value_by_comparables(target: TargetFinancials, peers: list[PeerCompany]) -> ComparableValuation:
    out = ComparableValuation(target=target.model_dump())

    def per_share(equity: float) -> float | None:
        """Multiples value the existing business (pre-money). New shares dilute that value only if the
        fresh-issue money is added back, so either divide by the pre-issue count or add proceeds and use the post-issue count."""
        if target.pre_issue_shares:
            return equity / target.pre_issue_shares
        if target.shares_outstanding:
            return (equity + (target.fresh_issue_inr or 0.0)) / target.shares_outstanding
        return None

    out.assumptions = [
        f"Target financials: {target.period} (from the DRHP's restated statements unless overridden).",
        "Peer multiples summarised by median; low/high are the minimum/maximum valid peer multiples.",
        "Multiples outside sane bounds (P/E 0-200x, EV/EBITDA 0-100x, P/S 0-50x) or from loss-making peers are excluded.",
        "Trailing multiples as supplied; no adjustment for growth, size or listing discount.",
        "Implied equity values are pre-money (the existing business, before fresh-issue proceeds).",
    ]
    if target.pre_issue_shares:
        out.assumptions.append("Price per share = pre-money equity / pre-issue shares.")
    elif target.shares_outstanding and target.fresh_issue_inr is not None:
        out.assumptions.append("Price per share = (pre-money equity + fresh-issue proceeds) / post-issue shares.")
    elif target.shares_outstanding:
        out.warnings.append("Only the post-issue share count was given: pre-money equity divided by post-issue shares understates the "
                            "value per share by the dilution from new shares. Enter the pre-issue share count, or the fresh-issue "
                            "proceeds (0 for a pure offer for sale).")
    if len(peers) < 3:
        out.warnings.append(f"Only {len(peers)} peer(s) supplied; at least 3-5 comparable listed companies are recommended.")
    if not peers:
        return out

    specs = [("P/E", "pe", "Profit after tax", target.pat, False),
             ("EV/EBITDA", "ev_ebitda", "EBITDA", target.ebitda, True),
             ("P/S", "ps", "Revenue", target.revenue, False)]
    for label, attr, metric, tval, is_ev in specs:
        vals, excl = {}, {}
        for p in peers:
            v = getattr(p, attr)
            lo, hi = _BOUNDS[label]
            if v is None:
                excl[p.name] = "not available"
            elif not lo < v < hi:
                excl[p.name] = f"{v:.1f}x outside {lo}-{hi}x"
            else:
                vals[p.name] = v
        r = MultipleResult(label, vals, excl, None, None, None, metric, tval)
        if not vals:
            r.note = "No valid peer multiples."
            out.results.append(r)
            continue
        r.low, r.median, r.high = min(vals.values()), median(vals.values()), max(vals.values())
        if tval is None:
            r.note = f"Target {metric.lower()} not available."
        elif tval <= 0:
            r.note = f"Target {metric.lower()} is not positive; this multiple cannot value it."
        else:
            adj = -(target.net_debt or 0.0) if is_ev else 0.0
            if is_ev and target.net_debt is None:
                r.note = "Net debt not available; EV treated as equity value (overstates equity if the company has debt)."
            r.implied_equity_low, r.implied_equity_mid, r.implied_equity_high = (m * tval + adj for m in (r.low, r.median, r.high))
            if per_share(0.0) is not None:
                r.implied_price_low, r.implied_price_mid, r.implied_price_high = (
                    per_share(e) for e in (r.implied_equity_low, r.implied_equity_mid, r.implied_equity_high))
        out.results.append(r)

    usable = [r for r in out.results if r.implied_equity_mid is not None]
    if usable:
        lows = [r.implied_equity_low for r in usable]
        mids = [r.implied_equity_mid for r in usable]
        highs = [r.implied_equity_high for r in usable]
        out.equity_range = (min(lows), median(mids), max(highs))
        if per_share(0.0) is not None:
            out.price_range = tuple(per_share(v) for v in out.equity_range)  # type: ignore[assignment]
        else:
            out.warnings.append("Share count not provided: implied price per share not computed.")
        out.assumptions.append("Overall range: lowest low and highest high across usable multiples; midpoint is the median of the multiples' midpoints.")
    return out
