"""Discounted cash flow valuation (free cash flow to the firm, Gordon growth terminal value).

All assumptions are explicit inputs; the base-year figures come from the DRHP and are
labelled as such, user assumptions are labelled as user inputs. Projection for year t:

    revenue_t = revenue_{t-1} x (1 + g_t)
    EBITDA_t  = revenue_t x EBITDA margin
    EBIT_t    = EBITDA_t - D&A_t           (D&A as % of revenue)
    NOPAT_t   = EBIT_t x (1 - tax rate)
    FCFF_t    = NOPAT_t + D&A_t - capex_t - change in working capital_t
    TV        = FCFF_N x (1 + g_terminal) / (WACC - g_terminal)
    EV        = sum FCFF_t / (1 + WACC)^t + TV / (1 + WACC)^N   (end-of-year discounting)
    Equity    = EV - net debt + fresh-issue proceeds;  price = equity / post-issue shares
"""

from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field, model_validator


class DCFAssumptions(BaseModel):
    base_revenue: float = Field(gt=0, description="Base-year revenue, absolute INR (from the DRHP)")
    base_period: str = "latest FY"
    revenue_growth: list[float] = Field(description="Growth per projection year, e.g. [0.15, 0.13, 0.11, 0.10, 0.09]")
    ebitda_margin: float = Field(ge=-1, le=1)
    tax_rate: float = Field(default=0.2517, ge=0, le=0.6, description="Default: Indian corporate rate under s.115BAA incl. surcharge/cess")
    da_pct_revenue: float = Field(default=0.03, ge=0, le=0.5)
    capex_pct_revenue: float = Field(default=0.04, ge=0, le=1)
    nwc_pct_incremental_revenue: float = Field(default=0.10, ge=-1, le=1, description="Working capital invested per rupee of revenue growth")
    wacc: float = Field(gt=0, lt=1)
    terminal_growth: float = Field(ge=-0.05, lt=0.15)
    net_debt: float = Field(default=0.0, description="Total debt - cash, absolute INR (from the DRHP)")
    shares_outstanding: float | None = Field(default=None, gt=0, description="Post-issue share count")
    fresh_issue_inr: float | None = Field(default=None, ge=0, description="Fresh-issue proceeds, absolute INR; added to equity so it matches the post-issue share count")
    input_sources: dict[str, str] = Field(default_factory=dict, description="field -> 'DRHP p.N' or 'user input'")

    @model_validator(mode="after")
    def _check(self) -> "DCFAssumptions":
        if not 1 <= len(self.revenue_growth) <= 15:
            raise ValueError("revenue_growth must cover 1-15 projection years")
        if self.terminal_growth >= self.wacc:
            raise ValueError("terminal_growth must be below WACC (otherwise the terminal value is infinite)")
        return self


@dataclass
class DCFYear:
    year: int
    revenue: float
    ebitda: float
    da: float
    ebit: float
    nopat: float
    capex: float
    delta_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float


@dataclass
class DCFResult:
    years: list[DCFYear]
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    price_per_share: float | None
    terminal_share_of_ev: float
    sensitivity_wacc_g: dict = field(default_factory=dict)
    sensitivity_growth: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _run(a: DCFAssumptions) -> tuple[list[DCFYear], float, float, float]:
    years, rev_prev = [], a.base_revenue
    for t, g in enumerate(a.revenue_growth, start=1):
        rev = rev_prev * (1 + g)
        ebitda = rev * a.ebitda_margin
        da = rev * a.da_pct_revenue
        ebit = ebitda - da
        nopat = ebit * (1 - a.tax_rate) if ebit > 0 else ebit  # no tax credit assumed on losses
        capex = rev * a.capex_pct_revenue
        dnwc = (rev - rev_prev) * a.nwc_pct_incremental_revenue
        fcff = nopat + da - capex - dnwc
        df = 1 / (1 + a.wacc) ** t
        years.append(DCFYear(t, rev, ebitda, da, ebit, nopat, capex, dnwc, fcff, df, fcff * df))
        rev_prev = rev
    pv_explicit = sum(y.pv_fcff for y in years)
    tv = years[-1].fcff * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
    pv_tv = tv * years[-1].discount_factor
    return years, pv_explicit, tv, pv_tv


def run_dcf(a: DCFAssumptions) -> DCFResult:
    years, pv_explicit, tv, pv_tv = _run(a)
    ev = pv_explicit + pv_tv
    equity = ev - a.net_debt + (a.fresh_issue_inr or 0.0)
    res = DCFResult(
        years=years, pv_explicit=pv_explicit, terminal_value=tv, pv_terminal=pv_tv, enterprise_value=ev,
        net_debt=a.net_debt, equity_value=equity,
        price_per_share=equity / a.shares_outstanding if a.shares_outstanding else None,
        terminal_share_of_ev=pv_tv / ev if ev else 0.0, assumptions=a.model_dump(),
    )
    if res.terminal_share_of_ev > 0.75:
        res.warnings.append(f"Terminal value is {res.terminal_share_of_ev:.0%} of enterprise value; the result is highly sensitive to WACC and terminal growth.")
    if years[-1].fcff <= 0:
        res.warnings.append("Final-year free cash flow is not positive, so the terminal value is not meaningful.")
    if a.shares_outstanding is None:
        res.warnings.append("Share count not provided: price per share not computed.")
    elif a.fresh_issue_inr is None:
        res.warnings.append("Fresh-issue proceeds not given: equity value is pre-money but is divided by the post-issue share count, "
                            "which understates the value per share if new shares are issued. Enter the proceeds (0 for a pure offer for sale).")
    res.sensitivity_wacc_g = sensitivity_wacc_growth(a)
    res.sensitivity_growth = sensitivity_revenue_growth(a)
    return res


def _value(a: DCFAssumptions) -> float:
    _, pv, _, pv_tv = _run(a)
    eq = pv + pv_tv - a.net_debt + (a.fresh_issue_inr or 0.0)
    return eq / a.shares_outstanding if a.shares_outstanding else eq


def sensitivity_wacc_growth(a: DCFAssumptions, wacc_steps=(-0.02, -0.01, 0, 0.01, 0.02), g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> dict:
    """Equity value (or price per share) for a grid of WACC x terminal growth."""
    waccs = [round(a.wacc + d, 4) for d in wacc_steps]
    gs = [round(a.terminal_growth + d, 4) for d in g_steps]
    grid = []
    for w in waccs:
        row = []
        for g in gs:
            row.append(None if g >= w or w <= 0 else _value(a.model_copy(update={"wacc": w, "terminal_growth": g})))
        grid.append(row)
    return {"metric": "price_per_share" if a.shares_outstanding else "equity_value", "wacc": waccs, "terminal_growth": gs, "values": grid}


def sensitivity_revenue_growth(a: DCFAssumptions, shifts=(-0.05, -0.025, 0, 0.025, 0.05)) -> dict:
    """Value when every year's revenue growth shifts by the given amount."""
    out = []
    for s in shifts:
        out.append({"shift": s, "value": _value(a.model_copy(update={"revenue_growth": [g + s for g in a.revenue_growth]}))})
    return {"metric": "price_per_share" if a.shares_outstanding else "equity_value", "points": out}
