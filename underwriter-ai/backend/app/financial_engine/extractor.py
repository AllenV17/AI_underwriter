"""Turn parsed statement tables into normalised, sourced financial values.

Every value keeps what was printed (`value_original`, `unit`), its rupee value
(`value_inr`), the page it came from and the row label it was read from. Derived
figures (EBITDA, total debt, free cash flow...) are computed in code, flagged
`derived=True` and carry their formula. Missing inputs stay missing: nothing is
replaced with zero.
"""

import re
from dataclasses import asdict, dataclass, field

from app.financial_engine.line_items import FIELD_LABELS, LINE_ITEMS
from app.financial_engine.statement_parser import StatementTable, norm_label
from app.financial_engine.units import to_inr


@dataclass
class FinancialValue:
    field: str
    label: str
    period: str
    value_original: float | None
    unit: str | None
    value_inr: float | None
    statement: str
    basis: str
    source_page: int | None
    source_label: str
    derived: bool = False
    formula: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationCheck:
    name: str
    period: str
    passed: bool | None  # None = could not be checked (inputs missing)
    detail: str


@dataclass
class FinancialData:
    periods: list[str]
    basis: str
    unit: str | None
    values: list[FinancialValue] = field(default_factory=list)
    checks: list[ValidationCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statement_pages: dict[str, list[int]] = field(default_factory=dict)

    def get(self, fld: str, period: str) -> FinancialValue | None:
        return next((v for v in self.values if v.field == fld and v.period == period), None)

    def inr(self, fld: str, period: str) -> float | None:
        v = self.get(fld, period)
        return v.value_inr if v else None

    def to_dict(self) -> dict:
        return {
            "periods": self.periods, "basis": self.basis, "unit": self.unit,
            "values": [v.to_dict() for v in self.values], "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings, "statement_pages": self.statement_pages,
        }


def _row_page(table: StatementTable, row_index: int) -> int | None:
    """Pages are parsed in order and rows appended in order; find the page by y-resets."""
    if len(table.pages) == 1:
        return table.pages[0]
    page_i, prev_y = 0, -1.0
    for i, r in enumerate(table.rows):
        if r.y < prev_y - 50:
            page_i = min(page_i + 1, len(table.pages) - 1)
        prev_y = r.y
        if i == row_index:
            return table.pages[page_i]
    return table.pages[0]


def map_fields(tables: dict[str, StatementTable]) -> FinancialData:
    """Map statement rows to canonical fields for every period."""
    # The P&L's periods define the reporting periods; other statements are matched by label.
    pl = tables.get("pl")
    periods = pl.periods if pl and pl.periods else next((t.periods for t in tables.values() if t.periods), [])
    basis = pl.basis if pl else next((t.basis for t in tables.values()), "unknown")
    unit = pl.unit if pl else next((t.unit for t in tables.values() if t.unit), None)
    data = FinancialData(periods=periods, basis=basis, unit=unit,
                         statement_pages={k: t.pages for k, t in tables.items()})
    for t in tables.values():
        data.warnings.extend(f"{t.statement}: {w}" for w in t.warnings)
        if t.basis != basis and t.basis != "unknown" and basis != "unknown":
            data.warnings.append(f"{t.statement} is {t.basis} but the P&L is {basis}")

    for li in LINE_ITEMS:
        t = tables.get(li.statement)
        if not t or not t.periods:
            continue
        labels = [norm_label(r.label) for r in t.rows]
        matched: list[int] = []
        for pat in li.patterns:
            for i, lab in enumerate(labels):
                if i in matched or not re.search(pat, lab) or any(re.search(x, lab) for x in li.exclude):
                    continue
                if all(v is None for v in t.rows[i].values):
                    continue
                matched.append(i)
                if not li.sum:
                    break
            # Single-value fields stop at the first pattern that matched; sum fields add every pattern.
            if matched and not li.sum:
                break
        if not matched:
            continue
        for ci, period in enumerate(t.periods):
            if period not in periods:
                continue
            vals = [t.rows[i].values[ci] for i in matched if ci < len(t.rows[i].values)]
            present = [v for v in vals if v is not None]
            if not present:
                continue
            original = sum(present) if li.sum else present[0]
            data.values.append(FinancialValue(
                field=li.field, label=li.label, period=period, value_original=original, unit=t.unit,
                value_inr=to_inr(original, t.unit) if t.unit and not li.field.startswith("eps") else
                (original if li.field.startswith("eps") else None),
                statement=li.statement, basis=t.basis, source_page=_row_page(t, matched[0]),
                source_label=" + ".join(t.rows[i].label for i in matched),
                formula=("sum of rows: " + "; ".join(t.rows[i].label for i in matched)) if li.sum and len(matched) > 1 else None,
            ))
    _derive(data)
    _validate(data)
    return data


def _add_derived(data: FinancialData, fld: str, period: str, inr: float, formula: str, page: int | None) -> None:
    unit = data.unit
    original = inr / _mult(unit) if unit else None
    data.values.append(FinancialValue(
        field=fld, label=FIELD_LABELS.get(fld, fld), period=period, value_original=original, unit=unit,
        value_inr=inr, statement="derived", basis=data.basis, source_page=page, source_label="derived",
        derived=True, formula=formula,
    ))


def _mult(unit: str) -> float:
    from app.financial_engine.units import MULTIPLIERS
    return MULTIPLIERS[unit]


def _derive(data: FinancialData) -> None:
    for p in data.periods:
        g = lambda f: data.inr(f, p)  # noqa: E731
        page = (data.get("profit_before_tax", p) or data.get("total_assets", p) or FinancialValue(
            "", "", p, None, None, None, "", "", None, "")).source_page
        if g("tax_expense") is None and g("tax_components") is not None:
            _add_derived(data, "tax_expense", p, g("tax_components"),
                         "Sum of tax component rows (current, deferred, earlier years); no total tax line printed",
                         data.get("tax_components", p).source_page)
        if g("profit_before_tax") is None and g("profit_before_exceptional_and_tax") is not None:
            _add_derived(data, "profit_before_tax", p, g("profit_before_exceptional_and_tax"),
                         "Profit before exceptional items and tax (no separate profit-before-tax line; no exceptional items shown)",
                         data.get("profit_before_exceptional_and_tax", p).source_page)
        pbt, fin, dep, oth = g("profit_before_tax"), g("finance_costs"), g("depreciation"), g("other_income")
        if pbt is not None and fin is not None:
            _add_derived(data, "ebit", p, pbt + fin, "Profit before tax + Finance costs", page)
            if dep is not None:
                ebitda = pbt + fin + dep - (oth or 0.0)
                _add_derived(data, "ebitda", p, ebitda,
                             "Profit before tax + Finance costs + Depreciation and amortisation"
                             + (" - Other income" if oth is not None else " (other income not found, not deducted)"), page)
        if g("total_equity") is None and g("equity_share_capital") is not None and g("other_equity") is not None:
            _add_derived(data, "total_equity", p, g("equity_share_capital") + g("other_equity"),
                         "Equity share capital + Other equity (no total equity line printed)",
                         data.get("equity_share_capital", p).source_page)
        if g("total_liabilities") is None and g("total_equity_and_liabilities") is not None and g("total_equity") is not None:
            _add_derived(data, "total_liabilities", p, g("total_equity_and_liabilities") - g("total_equity"),
                         "Total equity and liabilities - Total equity", page)
        debt = g("borrowings")
        if debt is not None:
            _add_derived(data, "total_debt", p, debt, "Sum of borrowings / debt securities / subordinated liabilities rows", page)
            cash = g("cash_and_equivalents")
            if cash is not None:
                _add_derived(data, "net_debt", p, debt - cash, "Total debt - Cash and cash equivalents", page)
        ocf, capex = g("operating_cash_flow"), g("capex")
        if ocf is not None and capex is not None:
            _add_derived(data, "free_cash_flow", p, ocf - abs(capex),
                         "Net cash from operating activities - |Purchase of property, plant and equipment|",
                         (data.get("operating_cash_flow", p) or data.get("capex", p)).source_page)
        pat, eps = g("profit_after_tax"), data.inr("eps_basic", p)
        if pat is not None and eps not in (None, 0):
            data.values.append(FinancialValue(
                field="weighted_shares", label=FIELD_LABELS["weighted_shares"], period=p, value_original=pat / eps,
                unit="shares", value_inr=pat / eps, statement="derived", basis=data.basis, source_page=page,
                source_label="derived", derived=True,
                formula="Profit after tax (₹) / Basic EPS (₹). Approximate; use the share count from the capital structure for valuation."))


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


def _validate(data: FinancialData) -> None:
    for p in data.periods:
        g = lambda f: data.inr(f, p)  # noqa: E731
        ta, tel = g("total_assets"), g("total_equity_and_liabilities")
        if ta is not None and tel is not None:
            data.checks.append(ValidationCheck("Total assets = Total equity and liabilities", p, _close(ta, tel, 0.002),
                                               f"{ta:,.0f} vs {tel:,.0f}"))
        else:
            data.checks.append(ValidationCheck("Total assets = Total equity and liabilities", p, None, "inputs not found"))
        pbt, tax, pat = g("profit_before_tax"), g("tax_expense"), g("profit_after_tax")
        if None not in (pbt, tax, pat):
            data.checks.append(ValidationCheck("Profit before tax - Tax = Profit after tax", p, _close(pbt - tax, pat, 0.02),
                                               f"{pbt - tax:,.0f} vs {pat:,.0f} (2% tolerance for exceptional items / associates)"))
        rev, oth, ti = g("revenue_from_operations"), g("other_income"), g("total_income")
        if None not in (rev, oth, ti):
            data.checks.append(ValidationCheck("Revenue + Other income = Total income", p, _close(rev + oth, ti, 0.002),
                                               f"{rev + oth:,.0f} vs {ti:,.0f}"))
    failed = [c for c in data.checks if c.passed is False]
    if failed:
        data.warnings.append(f"{len(failed)} accounting check(s) failed; review the flagged values before relying on them.")
