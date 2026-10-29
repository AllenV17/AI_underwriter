"""Synonym dictionary mapping statement line labels to canonical fields.

Each field lists regexes tried in priority order against the normalised row label
(lower-case, single spaces, no trailing punctuation). The first pattern that matches any
row wins; within a pattern the first matching row wins, except fields marked `sum`, which
add every matching row (e.g. borrowings that appear under both non-current and current).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineItem:
    field: str
    statement: str  # "pl", "bs" or "cf"
    patterns: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    sum: bool = False
    label: str = ""


LINE_ITEMS: list[LineItem] = [
    # --- Profit and loss ---
    LineItem("revenue_from_operations", "pl",
             (r"^total revenue from operations", r"^revenue from operations", r"^income from operations", r"^revenue from contracts? with customers"),
             label="Revenue from operations"),
    LineItem("other_income", "pl", (r"^other income",), label="Other income"),
    LineItem("total_income", "pl", (r"^total income", r"^total revenue$"), label="Total income"),
    LineItem("cost_of_materials", "pl", (r"^cost of (raw )?materials? consumed",), label="Cost of materials consumed"),
    LineItem("purchases_stock_in_trade", "pl", (r"^purchases? of (stock|traded goods)",), label="Purchases of stock-in-trade"),
    LineItem("change_in_inventories", "pl", (r"^changes? in inventor",), label="Changes in inventories"),
    LineItem("employee_benefits", "pl", (r"^employee benefits? expenses?",), label="Employee benefits expense"),
    LineItem("finance_costs", "pl", (r"^finance costs?", r"^interest expenses?$", r"^finance charges"), label="Finance costs"),
    LineItem("depreciation", "pl", (r"^depreciation", r"^amorti[sz]ation and depreciation"), label="Depreciation and amortisation"),
    LineItem("total_expenses", "pl", (r"^total expenses",), label="Total expenses"),
    LineItem("profit_before_tax", "pl",
             (r"^(restated )?(net )?profit\s*/?\s*\(?(loss)?\)?\s*before tax", r"before tax$", r"^profit before tax"),
             exclude=(r"exceptional", r"share of"), label="Profit before tax"),
    LineItem("profit_before_exceptional_and_tax", "pl", (r"profit\s*/?\s*\(?(loss)?\)?\s*before exceptional items? and tax",),
             label="Profit before exceptional items and tax"),
    LineItem("tax_expense", "pl", (r"^total tax expenses?", r"^tax expenses?", r"^total income tax expense", r"^income tax expense"),
             label="Tax expense"),
    LineItem("profit_after_tax", "pl",
             (r"^(restated )?(net )?profit\s*/?\s*\(?(loss)?\)?\s*for the (year|period)",
              r"^(restated )?profit\s*/?\s*\(?(loss)?\)?\s*after tax",
              r"^(restated )?(net )?profit\s*/?\s*\(?(loss)?\)?\s*(\([^)]*\))?$"),
             exclude=(r"attributable", r"other comprehensive", r"before", r"discontinued", r"continuing operations"),
             label="Profit after tax"),
    LineItem("tax_components", "pl", (r"^(current tax|income tax|deferred tax|tax (adjustments?|relating) (of|to) (earlier|prior) (years?|periods?))",),
             exclude=(r"relating to", r"other comprehensive", r"expense"), sum=True,
             label="Tax components (used when no total tax line)"),
    LineItem("eps_basic", "pl", (r"^basic( earnings per share)?( \(in ₹\)| \(₹\))?", r"^- basic", r"basic eps", r"^equity shares,? nominal value"),
             exclude=(r"weighted",), label="Basic EPS"),
    LineItem("eps_diluted", "pl", (r"^diluted( earnings per share)?", r"^- diluted", r"diluted eps"),
             exclude=(r"weighted",), label="Diluted EPS"),
    # --- Balance sheet ---
    LineItem("total_assets", "bs", (r"^total assets",), label="Total assets"),
    LineItem("total_current_assets", "bs", (r"^total current assets",), label="Total current assets"),
    LineItem("inventories", "bs", (r"^inventories",), label="Inventories"),
    LineItem("trade_receivables", "bs", (r"^trade receivables",), sum=True, label="Trade receivables"),
    LineItem("cash_and_equivalents", "bs", (r"^cash and cash equivalents?$", r"^cash and cash equivalents? \(", r"^cash and cash equivalents?"), label="Cash and cash equivalents"),
    LineItem("other_bank_balances", "bs", (r"^bank balances? other than",), label="Other bank balances"),
    LineItem("equity_share_capital", "bs", (r"^equity share capital", r"^share capital"), label="Equity share capital"),
    LineItem("total_equity", "bs",
             (r"^total equity$", r"^total equity \([a-z]\)$", r"^total equity attributable", r"^equity attributable to (the )?(equity )?(holders|owners|shareholders)",
              r"^total equity \(", r"^net worth"),
             exclude=(r"and liabilities",), label="Total equity"),
    LineItem("other_equity", "bs", (r"^other equity", r"^reserves and surplus"), label="Other equity / reserves"),
    LineItem("total_liabilities", "bs", (r"^total liabilities$",), label="Total liabilities"),
    LineItem("total_current_liabilities", "bs", (r"^total current liabilities",), label="Total current liabilities"),
    LineItem("borrowings", "bs", (r"^borrowings", r"^debt securities", r"^subordinated liabilities"),
             exclude=(r"lease",), sum=True, label="Borrowings (all)"),
    LineItem("total_equity_and_liabilities", "bs", (r"^total equity (and|&) liabilities", r"^total liabilities (and|&) equity"),
             label="Total equity and liabilities"),
    # --- Cash flow ---
    LineItem("operating_cash_flow", "cf",
             (r"^net cash (flows? )?(generated from|from|\(used in\)|used in|generated/\(used in\)|\(used in\)/generated from|from/\(used in\)).*operating activities",
              r"net cash.*operating activities", r"^net cash (flows? )?(generated|used)[a-z /()]* (from|in) operations$"),
             label="Net cash from operating activities"),
    LineItem("capex", "cf",
             (r"^(purchases?|acquisitions?|payments? for (the )?(purchases?|acquisitions?)) of (property, plant and equipment|property plant and equipment|ppe|fixed assets|tangible)",
              r"^(purchases?|acquisitions?) of .*property,? plant and equipment"),
             label="Purchase of property, plant and equipment"),
    LineItem("investing_cash_flow", "cf", (r"net cash.*investing activities",), label="Net cash from investing activities"),
    LineItem("financing_cash_flow", "cf", (r"net cash.*financing activities",), label="Net cash from financing activities"),
]

STATEMENT_NAMES = {"pl": "Statement of profit and loss", "bs": "Statement of assets and liabilities", "cf": "Statement of cash flows"}
FIELD_LABELS = {li.field: li.label for li in LINE_ITEMS} | {
    "ebitda": "EBITDA (derived)", "ebit": "EBIT (derived)", "total_debt": "Total debt (derived)",
    "net_debt": "Net debt (derived)", "free_cash_flow": "Free cash flow (derived)",
    "weighted_shares": "Weighted average shares (derived from PAT / basic EPS)",
}
