"""Locate the primary financial statements in a DRHP and rebuild their tables.

DRHP statements are mostly borderless, so ruled-table extractors split or merge their
columns. Instead this module rebuilds rows from word positions (PyMuPDF words):

1. Lines are formed from words sharing a baseline.
2. In each line, numeric tokens to the right of the last word are its values.
3. Value columns are found by clustering the right edges of numbers (figures are
   right-aligned), so a "Note" column and wrapped labels do not shift the periods.
4. Column headers (periods such as "March 31, 2024") are read from words above the data.

Statement pages are found by scoring every page on how many of that statement's
expected line items it contains, which separates the statement from the many note pages
that repeat its title. Only pure functions here; no I/O beyond reading the PDF pages.
"""

import re
from dataclasses import dataclass, field

import pymupdf

from app.financial_engine.line_items import LINE_ITEMS
from app.financial_engine.units import detect_unit, is_number_token, parse_number

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_ALPHA = re.compile(r"[A-Za-z]")


@dataclass
class Row:
    label: str
    values: list[float | None]
    raw: list[str]
    y: float


@dataclass
class StatementTable:
    statement: str  # pl / bs / cf
    pages: list[int]  # 1-based
    basis: str  # consolidated / standalone / unknown
    unit: str | None
    unit_caption: str | None
    periods: list[str]
    rows: list[Row] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- line reconstruction

def _lines(page: "pymupdf.Page", y_tol: float = 2.8) -> list[list[tuple]]:
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if lines and abs(((lines[-1][0][1] + lines[-1][0][3]) / 2) - yc) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w[0]) for ln in lines]


_YEAR = re.compile(r"^(19[89]\d|20[0-4]\d)[,.]?$")
_DATE_WORD = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.,]?$|^\d{1,2}(st|nd|rd|th)?,?$", re.I)


def _is_header_line(line: list[tuple]) -> bool:
    """A line of period headings, e.g. '31 March 2024  31 March 2023' or 'Fiscal 2024  Fiscal 2023'."""
    toks = [w[4] for w in line]
    years = [t for t in toks if _YEAR.match(t)]
    if not years:
        return False
    nums = [t for t in toks if is_number_token(t) or _YEAR.match(t)]
    # Every number on the line is a year or a day-of-month, and the line has no decimals/commas figures.
    return all(_YEAR.match(t) or re.fullmatch(r"\d{1,2},?", t) for t in nums) and not any(
        "," in t and not t.endswith(",") for t in nums)


def _split_line(line: list[tuple]) -> tuple[str, list[tuple]]:
    """Label text and the numeric value tokens to the right of the last word."""
    if _is_header_line(line):
        return " ".join(w[4] for w in line), []
    last_alpha = max((i for i, w in enumerate(line) if _ALPHA.search(w[4])), default=-1)
    label = " ".join(w[4] for w in line[: last_alpha + 1])
    values = [w for w in line[last_alpha + 1:] if is_number_token(w[4])]
    return label, values


def norm_label(label: str) -> str:
    t = label.lower().replace("’", "'").replace("–", "-")
    t = re.sub(r"\s+", " ", t)
    # Strip list numbering: "(a) ", "iv. ", "1. ", "vii ", "iv expenses i) ..." -> label proper.
    for _ in range(2):
        t = re.sub(r"^[\(\[]?([ivx]{1,5}|[a-h]|\d{1,2}(\.\d+)?)[\)\].]?\s+(?=[a-z(])", "", t)
    # A heading merged with its first numbered sub-item ("expenses i) finance cost"): keep the item.
    m = re.search(r"\s\(?[ivx]{1,4}\)\s+(?=[a-z])", t)
    if m:
        t = t[m.end():]
    return t.strip(" :.-*")


def _cluster(xs: list[float], gap: float = 22.0) -> list[list[float]]:
    xs = sorted(xs)
    out: list[list[float]] = []
    for x in xs:
        if out and x - out[-1][-1] <= gap:
            out[-1].append(x)
        else:
            out.append([x])
    return out


_MONTH_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?", re.I)


def parse_period(text: str) -> str | None:
    """'March 31, 2024' -> 'FY2024'; 'December 31, 2022' -> 'Dec 2022'; 'Fiscal 2023' -> 'FY2023'; '2023-24' -> 'FY2024'."""
    t = text.replace(",", " ")
    m = re.search(r"(?:fiscal|fy)\s*'?(\d{2,4})\b", t, re.I)
    if m:
        y = int(m.group(1))
        return f"FY{y + 2000 if y < 100 else y}"
    m = re.search(r"\b(20\d{2})\s*-\s*(\d{2,4})\b", t)
    if m:
        y2 = int(m.group(2))
        return f"FY{y2 + 2000 if y2 < 100 else y2}"
    mon = _MONTH_RE.search(t)
    yr = re.search(r"\b(20\d{2}|19\d{2})\b", t)
    if yr and mon:
        mnum = MONTHS[mon.group(1).lower()[:3]]
        y = int(yr.group(1))
        return f"FY{y}" if mnum == 3 else f"{mon.group(1)[:3].title()} {y}"
    if yr:
        return f"FY{yr.group(1)}"
    return None


def parse_table(page: "pymupdf.Page", min_col_rows: int = 3) -> tuple[list[str], list[Row], list[str]]:
    """Rebuild (period labels, rows, warnings) from one statement page."""
    lines = _lines(page)
    parsed = []
    for ln in lines:
        label, vals = _split_line(ln)
        parsed.append((ln, label, vals))
    data = [(ln, label, vals) for ln, label, vals in parsed if vals]
    warnings: list[str] = []
    if len(data) < 4:
        return [], [], ["too few numeric rows"]

    clusters = [c for c in _cluster([w[2] for _, _, vals in data for w in vals]) if len(c) >= min_col_rows]
    centers = [sum(c) / len(c) for c in clusters]  # right-edge centres

    # The table body starts at the first line carrying most of the columns; stray figures in a
    # title area (e.g. "0.00" artefacts) must not be mistaken for the first data row.
    widest = max(len(vals) for _, _, vals in data)
    body = [ln for ln, _, vals in data if len(vals) >= max(2, int(0.6 * widest))]
    first_data_y = min(ln[0][1] for ln in body) if body else min(ln[0][1] for ln, _, _ in data)
    header_words = [w for ln, _, _ in parsed for w in ln if w[3] <= first_data_y + 1 and w[1] > first_data_y - 160]

    def col_of(x1: float) -> int | None:
        if not centers:
            return None
        i = min(range(len(centers)), key=lambda k: abs(centers[k] - x1))
        return i if abs(centers[i] - x1) <= 30 else None

    # Header text per column: words whose right part falls near the column's right edge zone.
    head_txt = []
    for i, cx in enumerate(centers):
        left = centers[i - 1] + 8 if i > 0 else cx - 110
        ws = [w for w in header_words if left < (w[0] + w[2]) / 2 <= cx + 12]
        ws.sort(key=lambda w: (round(w[1]), w[0]))
        head_txt.append(" ".join(w[4] for w in ws))

    # Drop a "Note" column: header says note, or it holds only small whole numbers.
    keep = []
    for i, c in enumerate(clusters):
        toks = [w[4] for _, _, vals in data for w in vals if col_of(w[2]) == i]
        small_ints = all(re.fullmatch(r"\d{1,3}(\.\d{1,2})?[A-Za-z]?", t) and "," not in t for t in toks) and toks
        is_note = "note" in head_txt[i].lower() or (small_ints and i == 0 and len(clusters) > 1
                                                     and all(float(re.sub(r'[^\d.]', '', t) or 0) < 100 for t in toks))
        if not is_note:
            keep.append(i)
    periods = []
    for i in keep:
        p = parse_period(head_txt[i])
        if p is None:
            warnings.append(f"period header not recognised for column {len(periods) + 1}: '{head_txt[i][:60]}'")
            p = f"col{len(periods) + 1}"
        periods.append(p)

    rows: list[Row] = []
    pending = ""
    for ln, label, vals in parsed:
        if not vals:
            pending = label if label and len(label) < 120 else ""
            continue
        vec: list[float | None] = [None] * len(keep)
        raw = [""] * len(keep)
        for w in vals:
            ci = col_of(w[2])
            if ci is not None and ci in keep:
                k = keep.index(ci)
                vec[k], raw[k] = parse_number(w[4]), w[4]
        full = label
        if pending and (not label or label[:1].islower() or pending.rstrip().endswith(("/", "(", "and", "-", ","))):
            full = f"{pending} {label}".strip()
        rows.append(Row(label=full, values=vec, raw=raw, y=ln[0][1]))
        pending = ""
    return periods, rows, warnings


# ---------------------------------------------------------------- locating statements

_HEADINGS = {
    "pl": r"statement of (restated )?(consolidated |standalone )?profit (and|&) loss",
    "bs": r"statement of (restated )?(consolidated |standalone )?assets and liabilities|balance sheet",
    "cf": r"statement of (restated )?(consolidated |standalone )?cash flows?|cash flow statement",
}
_NOT_STATEMENT = re.compile(r"notes? (forming part|to (the )?(restated )?(consolidated )?financial)|examination report|"
                            r"significant accounting policies", re.I)
_TERMINAL = {"pl": r"(earnings per|diluted|basic)", "bs": r"total equity and liabilities|total liabilities and equity",
             "cf": r"cash and cash equivalents at the end|closing cash"}


def _score_page(text_top: str, labels: list[str], statement: str) -> float:
    items = [li for li in LINE_ITEMS if li.statement == statement]
    hits = sum(1 for li in items if any(re.search(p, lab) for lab in labels for p in li.patterns))
    score = float(hits)
    if re.search(_HEADINGS[statement], text_top, re.I):
        score += 3
    if _NOT_STATEMENT.search(text_top[:400]):
        score -= 6
    return score


def _basis(text_top: str) -> str:
    t = text_top.lower()
    if "consolidated" in t:
        return "consolidated"
    if "standalone" in t:
        return "standalone"
    return "unknown"


def locate_and_parse(pdf_path: str, candidate_pages: list[int], all_pages: int) -> dict[str, StatementTable]:
    """Find and parse the P&L, balance sheet and cash-flow statements.

    `candidate_pages` are 1-based pages to search first (the restated-financials chapter);
    if a statement is not found there, the whole document is searched.
    """
    pymupdf.TOOLS.mupdf_display_errors(False)
    out: dict[str, StatementTable] = {}
    with pymupdf.open(pdf_path) as doc:
        cache: dict[int, tuple[str, list[str]]] = {}

        def info(n: int) -> tuple[str, list[str]]:
            if n not in cache:
                pg = doc[n - 1]
                top = pg.get_text()[:700]
                labels = [norm_label(_split_line(ln)[0]) for ln in _lines(pg)]
                cache[n] = (top, [lab for lab in labels if lab])
            return cache[n]

        for st in ("pl", "bs", "cf"):
            best = None
            for pool in (candidate_pages, range(1, all_pages + 1)):
                scored = []
                for n in pool:
                    top, labels = info(n)
                    s = _score_page(top, labels, st)
                    if s >= 4:
                        scored.append((s, _basis(top) == "consolidated", -n, n))
                if scored:
                    # Highest score; prefer consolidated; then the earliest page.
                    scored.sort(reverse=True)
                    best = scored[0][3]
                    break
            if best is None:
                continue
            # A statement can start on the page before its best-scoring page (e.g. cash flows whose
            # operating section is on page n-1 and investing/financing on page n).
            if best > 1:
                ptop, plabels = info(best - 1)
                ocf = re.compile(r"net cash.*operating activities|cash flows? from operating activities")
                if (re.search(_HEADINGS[st], ptop, re.I) and not _NOT_STATEMENT.search(ptop[:400])
                        and _score_page(ptop, plabels, st) >= 4):
                    best -= 1
                elif (st == "cf" and not any(re.search(r"net cash.*operating activities", lab) for lab in info(best)[1])
                        and any(ocf.search(lab) for lab in plabels) and not _NOT_STATEMENT.search(ptop[:400])):
                    best -= 1  # operating section on the previous page, printed without a heading
            pages = [best]
            # Follow continuation pages until the statement's terminal line appears.
            for nxt in (best + 1, best + 2):
                top, labels = info(pages[-1])
                if any(re.search(_TERMINAL[st], lab) for lab in labels) or nxt > all_pages:
                    break
                ntop, nlabels = info(nxt)
                if _score_page(ntop, nlabels, st) >= 1 and not _NOT_STATEMENT.search(ntop[:400]):
                    pages.append(nxt)
                else:
                    break
            table = _parse_pages(doc, pages, st, info(best)[0])
            out[st] = table
    return out


def _parse_pages(doc: "pymupdf.Document", pages: list[int], st: str, top: str) -> StatementTable:
    unit = detect_unit(top) or detect_unit(doc[pages[0] - 1].get_text())
    periods: list[str] = []
    rows: list[Row] = []
    warnings: list[str] = []
    for n in pages:
        p, r, w = parse_table(doc[n - 1])
        if not periods and p:
            periods = p
        elif p and len(p) != len(periods):
            warnings.append(f"page {n}: {len(p)} columns vs {len(periods)} on the first page")
        rows.extend(r)
        warnings.extend(f"page {n}: {x}" for x in w)
    if unit is None:
        warnings.append("currency unit caption not found; values cannot be converted to rupees")
    return StatementTable(
        statement=st, pages=pages, basis=_basis(top), unit=unit[0] if unit else None,
        unit_caption=unit[1] if unit else None, periods=periods, rows=rows, warnings=warnings,
    )
