"""Deterministic financial ratio engine.

Every ratio is computed in Python from extracted values, never by the LLM. Each result
carries its formula, inputs (with source pages), period and a rule-based interpretation.
When an input is missing the ratio is returned with `value=None` and the reason.

Balance-sheet averages: returns use period-end balances (DRHPs give few periods; using
averages would lose the earliest year). This is stated in each formula.

Financial companies (NBFCs, HFCs, banks): interest is their cost of goods, so EBITDA,
interest coverage and current/quick ratios are not meaningful; they are reported as
not applicable with that reason.
"""

from dataclasses import asdict, dataclass, field

from app.financial_engine.extractor import FinancialData

STUB_NOTE = "Interim (stub) period: the flow figure covers less than a full year and is not annualised."
# Ratios that divide a period flow (profit, revenue, cash flow) by a balance: understated for stub periods.
FLOW_OVER_STOCK = {"Return on Equity", "Return on Capital Employed", "Return on Assets", "Asset Turnover",
                   "Inventory Turnover", "Receivables Turnover", "Working Capital Turnover", "Operating Cash Flow to Debt",
                   "Net Debt to EBITDA"}


@dataclass
class RatioInput:
    name: str
    value: float | None
    source_page: int | None
    derived: bool = False


@dataclass
class RatioResult:
    name: str
    category: str
    formula: str
    period: str
    value: float | None
    unit: str  # "%", "x", "days", "INR"
    inputs: list[RatioInput] = field(default_factory=list)
    interpretation: str = ""
    available: bool = True
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def is_financial_company(data: FinancialData) -> bool:
    """Lender detection: finance costs above 35% of revenue, or interest income reported as revenue."""
    for p in data.periods:
        rev, fin = data.inr("revenue_from_operations", p), data.inr("finance_costs", p)
        if rev and fin and fin / rev > 0.35:
            return True
    return False


def _band(v: float, bands: list[tuple[float, str]], default: str) -> str:
    for limit, text in bands:
        if v < limit:
            return text
    return default


class RatioEngine:
    def __init__(self, data: FinancialData, price_per_share: float | None = None,
                 shares_outstanding: float | None = None) -> None:
        self.d = data
        self.price = price_per_share
        self.shares = shares_outstanding
        self.financial_co = is_financial_company(data)

    # ------------------------------------------------------------------ helpers
    def _in(self, fld: str, p: str, name: str | None = None) -> RatioInput:
        v = self.d.get(fld, p)
        return RatioInput(name or (v.label if v else fld), v.value_inr if v else None,
                          v.source_page if v else None, v.derived if v else False)

    def _ratio(self, name, cat, formula, p, num: RatioInput, den: RatioInput, unit="%", interp=None,
               extra: list[RatioInput] | None = None, na: str | None = None) -> RatioResult:
        inputs = [num, den] + (extra or [])
        if na:
            return RatioResult(name, cat, formula, p, None, unit, inputs, available=False, reason=na)
        missing = [i.name for i in inputs if i.value is None]
        if missing:
            return RatioResult(name, cat, formula, p, None, unit, inputs, available=False,
                               reason=f"Not calculated: {', '.join(missing)} not found in the extracted statements.")
        if den.value == 0:
            return RatioResult(name, cat, formula, p, None, unit, inputs, available=False, reason="Denominator is zero.")
        v = num.value / den.value * (100 if unit == "%" else 1)
        text = interp(v) if interp else ""
        return RatioResult(name, cat, formula, p, round(v, 4), unit, inputs, text)

    # ------------------------------------------------------------------ ratios
    def compute(self) -> list[RatioResult]:
        out: list[RatioResult] = []
        for p in self.d.periods:
            out += self.profitability(p) + self.liquidity(p) + self.leverage(p) + self.efficiency(p)
            out += self.cash_flow(p) + self.valuation(p)
        for r in out:
            if r.available and not r.period.startswith("FY") and r.name in FLOW_OVER_STOCK:
                r.interpretation = f"{r.interpretation} {STUB_NOTE}".strip()
        return out

    def profitability(self, p: str) -> list[RatioResult]:
        rev = self._in("revenue_from_operations", p, "Revenue from operations")
        out = []
        # Gross profit = revenue - (materials + purchases + change in inventories); manufacturers/traders only.
        mats = [self.d.inr(f, p) for f in ("cost_of_materials", "purchases_stock_in_trade", "change_in_inventories")]
        if any(m is not None for m in mats) and rev.value is not None:
            cogs = sum(m for m in mats if m is not None)
            gp = RatioInput("Gross profit (Revenue - materials, purchases, inventory change)", rev.value - cogs,
                            self.d.get("revenue_from_operations", p).source_page, True)
            out.append(self._ratio("Gross Profit Margin", "Profitability",
                                   "(Revenue - Cost of materials - Purchases of stock-in-trade - Change in inventories) / Revenue",
                                   p, gp, rev, interp=lambda v: _band(v, [(20, "Thin gross margin."), (40, "Moderate gross margin.")], "High gross margin.")))
        else:
            out.append(RatioResult("Gross Profit Margin", "Profitability", "Gross profit / Revenue", p, None, "%", [rev],
                                   available=False, reason="No cost-of-goods lines (materials / purchases / inventory change) in the statement; common for services companies and lenders."))
        na = "Not meaningful for a lender: interest expense is its main operating cost." if self.financial_co else None
        out.append(self._ratio("EBITDA Margin", "Profitability", "EBITDA / Revenue from operations (EBITDA derived: PBT + finance costs + D&A - other income)",
                               p, self._in("ebitda", p, "EBITDA (derived)"), rev, na=na,
                               interp=lambda v: _band(v, [(0, "Negative EBITDA: operations are loss-making."), (10, "Low operating margin."), (20, "Moderate operating margin.")], "Strong operating margin.")))
        out.append(self._ratio("Operating Profit Margin", "Profitability", "EBIT / Revenue from operations (EBIT = PBT + finance costs)",
                               p, self._in("ebit", p, "EBIT (derived)"), rev, na=na,
                               interp=lambda v: _band(v, [(0, "Operating loss."), (10, "Low operating margin.")], "Healthy operating margin.")))
        out.append(self._ratio("Net Profit Margin", "Profitability", "Profit after tax / Revenue from operations",
                               p, self._in("profit_after_tax", p, "Profit after tax"), rev,
                               interp=lambda v: _band(v, [(0, "Loss-making in this period."), (5, "Thin net margin."), (15, "Moderate net margin.")], "High net margin.")))
        roe_i = lambda v: _band(v, [(0, "Negative return on equity."), (10, "Below a typical cost of equity (~12-14% in India)."), (15, "Around the cost of equity.")], "Above the typical cost of equity.")  # noqa: E731
        out.append(self._ratio("Return on Equity", "Profitability", "Profit after tax / Total equity (period-end)",
                               p, self._in("profit_after_tax", p), self._in("total_equity", p, "Total equity"), interp=roe_i))
        ce = None
        te, debt = self.d.inr("total_equity", p), self.d.inr("total_debt", p)
        if te is not None:
            ce = RatioInput("Capital employed (Total equity + Total debt)", te + (debt or 0.0),
                            self.d.get("total_equity", p).source_page, True)
        out.append(self._ratio("Return on Capital Employed", "Profitability",
                               "EBIT / (Total equity + Total debt), period-end" + ("" if debt is not None else "; no borrowings found, debt taken as nil"),
                               p, self._in("ebit", p, "EBIT (derived)"), ce or RatioInput("Capital employed", None, None), na=na,
                               interp=lambda v: _band(v, [(10, "Low return on capital employed."), (18, "Moderate return on capital.")], "Strong return on capital.")))
        out.append(self._ratio("Return on Assets", "Profitability", "Profit after tax / Total assets (period-end)",
                               p, self._in("profit_after_tax", p), self._in("total_assets", p, "Total assets"),
                               interp=lambda v: _band(v, [(0, "Negative return on assets."), (2, "Low (typical for lenders)."), (8, "Moderate.")], "High return on assets.")))
        return out

    def liquidity(self, p: str) -> list[RatioResult]:
        na = "Not meaningful for a lender, whose balance sheet is not split into current and non-current." if self.financial_co else None
        ca, cl = self._in("total_current_assets", p, "Total current assets"), self._in("total_current_liabilities", p, "Total current liabilities")
        inv = self._in("inventories", p, "Inventories")
        out = [self._ratio("Current Ratio", "Liquidity", "Total current assets / Total current liabilities", p, ca, cl, "x", na=na,
                           interp=lambda v: _band(v, [(1, "Current liabilities exceed current assets."), (1.5, "Adequate short-term liquidity.")], "Comfortable short-term liquidity."))]
        quick = None
        if ca.value is not None:
            quick = RatioInput("Current assets - Inventories", ca.value - (inv.value or 0.0), ca.source_page, True)
        out.append(self._ratio("Quick Ratio", "Liquidity", "(Total current assets - Inventories) / Total current liabilities",
                               p, quick or RatioInput("Current assets - Inventories", None, None), cl, "x", na=na,
                               interp=lambda v: _band(v, [(1, "Relies on inventory to meet short-term obligations.")], "Liquid assets cover current liabilities.")))
        out.append(self._ratio("Cash Ratio", "Liquidity", "Cash and cash equivalents / Total current liabilities", p,
                               self._in("cash_and_equivalents", p, "Cash and cash equivalents"), cl, "x", na=na,
                               interp=lambda v: _band(v, [(0.2, "Low cash cover.")], "Reasonable cash cover.")))
        return out

    def leverage(self, p: str) -> list[RatioResult]:
        debt = self._in("total_debt", p, "Total debt (derived)")
        out = [self._ratio("Debt-to-Equity", "Leverage", "Total debt / Total equity", p, debt, self._in("total_equity", p), "x",
                           interp=lambda v: (_band(v, [(4, "Moderate leverage for a lender."), (7, "High but common leverage for a lender.")], "Very high leverage.")
                                             if self.financial_co else _band(v, [(0.5, "Low leverage."), (1.5, "Moderate leverage.")], "High leverage.")))]
        out.append(self._ratio("Debt-to-Assets", "Leverage", "Total debt / Total assets", p, debt, self._in("total_assets", p), "x",
                               interp=lambda v: _band(v, [(0.3, "Assets are mostly funded by equity and operating liabilities.")], "A large share of assets is debt-funded.")))
        na = "Not meaningful for a lender: interest expense is its main operating cost." if self.financial_co else None
        out.append(self._ratio("Interest Coverage Ratio", "Leverage", "EBIT / Finance costs", p, self._in("ebit", p, "EBIT (derived)"),
                               self._in("finance_costs", p, "Finance costs"), "x", na=na,
                               interp=lambda v: _band(v, [(1.5, "Weak: earnings barely cover interest."), (3, "Adequate interest cover.")], "Comfortable interest cover.")))
        out.append(self._ratio("Net Debt to EBITDA", "Leverage", "(Total debt - Cash and cash equivalents) / EBITDA", p,
                               self._in("net_debt", p, "Net debt (derived)"), self._in("ebitda", p, "EBITDA (derived)"), "x", na=na,
                               interp=lambda v: _band(v, [(0, "Net cash position."), (2, "Low net leverage."), (3.5, "Moderate net leverage.")], "High net leverage.")))
        return out

    def efficiency(self, p: str) -> list[RatioResult]:
        rev = self._in("revenue_from_operations", p, "Revenue from operations")
        out = [self._ratio("Asset Turnover", "Efficiency", "Revenue from operations / Total assets (period-end)", p, rev, self._in("total_assets", p), "x",
                           interp=lambda v: _band(v, [(0.5, "Asset-heavy business (low turnover)."), (1.5, "Moderate asset turnover.")], "High asset turnover."))]
        cogs_vals = [self.d.inr(f, p) for f in ("cost_of_materials", "purchases_stock_in_trade", "change_in_inventories")]
        cogs = sum(v for v in cogs_vals if v is not None) if any(v is not None for v in cogs_vals) else None
        out.append(self._ratio("Inventory Turnover", "Efficiency", "Cost of goods sold (materials + purchases + inventory change) / Inventories (period-end)",
                               p, RatioInput("Cost of goods sold", cogs, rev.source_page, True), self._in("inventories", p, "Inventories"), "x",
                               interp=lambda v: f"Inventory turns over about {v:.1f} times a year (≈{365 / v:.0f} days)." if v else ""))
        out.append(self._ratio("Receivables Turnover", "Efficiency", "Revenue from operations / Trade receivables (period-end)", p, rev,
                               self._in("trade_receivables", p, "Trade receivables"), "x",
                               interp=lambda v: f"Receivables are collected about {v:.1f} times a year (≈{365 / v:.0f} days of sales outstanding)." if v else ""))
        wc = None
        ca, cl = self.d.inr("total_current_assets", p), self.d.inr("total_current_liabilities", p)
        if ca is not None and cl is not None:
            wc = RatioInput("Working capital (Current assets - Current liabilities)", ca - cl,
                            self.d.get("total_current_assets", p).source_page, True)
        na = "Not meaningful for a lender." if self.financial_co else None
        out.append(self._ratio("Working Capital Turnover", "Efficiency", "Revenue from operations / (Current assets - Current liabilities)",
                               p, rev, wc or RatioInput("Working capital", None, None), "x", na=na,
                               interp=lambda v: "Negative working capital: suppliers and customers fund operations." if v < 0 else f"Each rupee of working capital supports ₹{v:.1f} of revenue."))
        return out

    def cash_flow(self, p: str) -> list[RatioResult]:
        ocf, fcf = self.d.get("operating_cash_flow", p), self.d.get("free_cash_flow", p)
        out = []
        for name, v, formula in (("Operating Cash Flow", ocf, "Net cash from operating activities (as reported)"),
                                 ("Free Cash Flow", fcf, "Operating cash flow - Purchase of property, plant and equipment")):
            if v is None:
                out.append(RatioResult(name, "Cash flow", formula, p, None, "INR", [], available=False,
                                       reason="Not found in the extracted cash-flow statement."))
            else:
                text = "Positive: operations generate cash." if v.value_inr > 0 else "Negative: operations consume cash."
                if self.financial_co and name == "Operating Cash Flow" and v.value_inr < 0:
                    text = "Negative, as is usual for a growing lender: loan disbursements are operating outflows."
                out.append(RatioResult(name, "Cash flow", formula, p, v.value_inr, "INR",
                                       [RatioInput(v.label, v.value_inr, v.source_page, v.derived)], text))
        out.append(self._ratio("Operating Cash Flow to Debt", "Cash flow", "Net cash from operating activities / Total debt", p,
                               self._in("operating_cash_flow", p, "Operating cash flow"), self._in("total_debt", p, "Total debt"), "x",
                               interp=lambda v: _band(v, [(0, "Operating cash flow is negative."), (0.2, "Low debt cover from operating cash.")], "Operating cash covers a meaningful share of debt.")))
        return out

    def valuation(self, p: str) -> list[RatioResult]:
        eps = self.d.get("eps_basic", p)
        out = [RatioResult("EPS (basic)", "Valuation", "As reported in the restated statement of profit and loss", p,
                           eps.value_inr if eps else None, "INR", [RatioInput("Basic EPS", eps.value_inr, eps.source_page)] if eps else [],
                           "" if eps else "", available=eps is not None, reason=None if eps else "Basic EPS not found.")]
        if self.price is None:
            reason = "Requires a price per share (enter the issue price / price band to compute)."
            for n, f in (("P/E", "Price per share / Basic EPS"), ("EV/EBITDA", "(Market cap + Total debt - Cash) / EBITDA"),
                         ("Price-to-Sales", "Market cap / Revenue from operations"), ("Price-to-Book", "Market cap / Total equity")):
                out.append(RatioResult(n, "Valuation", f, p, None, "x", [], available=False, reason=reason))
            return out
        out.append(self._ratio("P/E", "Valuation", "Price per share / Basic EPS", p, RatioInput("Price per share (user input)", self.price, None),
                               RatioInput("Basic EPS", eps.value_inr if eps else None, eps.source_page if eps else None), "x"))
        if self.shares is None:
            reason = "Requires the share count (post-issue shares) to compute market capitalisation."
            for n in ("EV/EBITDA", "Price-to-Sales", "Price-to-Book"):
                out.append(RatioResult(n, "Valuation", "", p, None, "x", [], available=False, reason=reason))
            return out
        mcap = RatioInput("Market capitalisation (price x shares, user inputs)", self.price * self.shares, None, True)
        nd = self.d.inr("net_debt", p)
        ev = RatioInput("Enterprise value (market cap + net debt)", mcap.value + (nd or 0.0), None, True)
        na = "Not meaningful for a lender." if self.financial_co else None
        out.append(self._ratio("EV/EBITDA", "Valuation", "(Market cap + Total debt - Cash) / EBITDA", p, ev, self._in("ebitda", p, "EBITDA (derived)"), "x", na=na))
        out.append(self._ratio("Price-to-Sales", "Valuation", "Market cap / Revenue from operations", p, mcap, self._in("revenue_from_operations", p), "x"))
        out.append(self._ratio("Price-to-Book", "Valuation", "Market cap / Total equity", p, mcap, self._in("total_equity", p), "x"))
        return out
