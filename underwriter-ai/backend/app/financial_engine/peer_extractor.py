"""Read the listed-peer comparison table from the "Basis for Offer Price" chapter.

The table ("Name of the Company | Revenue | Face value | P/E | EPS | RoNW | NAV") is
borderless, so it is rebuilt from word positions with the same line and column logic as
the financial statements. Only figures printed in the DRHP are returned, each with its
page. Wrapped company names are rebuilt on a best-effort basis, so the result is a
suggestion for the user to verify against the cited page, never an automatic input; a DRHP that has not yet filled the table ("[●]") or states it has no comparable
listed peers yields an empty list with the reason. Nothing is guessed or fetched.
"""

import re
from dataclasses import asdict, dataclass, field

import pymupdf

from app.financial_engine.statement_parser import _cluster, _lines, _split_line
from app.financial_engine.units import detect_unit, parse_number, to_inr

# Header keyword -> field, tried in order against each column's header text.
COLUMNS: list[tuple[str, str]] = [
    (r"face\s*value", "face_value"),
    (r"p\s*/\s*e", "pe"),
    (r"eps.*dilut|dilut.*eps", "eps_diluted"),
    (r"eps|earnings per share", "eps_basic"),
    (r"ronw|return on net\s*worth", "ronw_pct"),
    (r"nav|net asset", "nav_per_share"),
    (r"revenue|total income|turnover", "revenue"),
]
_NAME_WORDS = re.compile(r"[A-Za-z]{3,}")
_TAIL = re.compile(r"^(limited|ltd\.?|\(\d\))", re.I)
_HEADER_HINT = re.compile(r"name of the (company|peer)|listed (industry )?peers?", re.I)
_LEAK = re.compile(r"\b(eps|p/\s*e|value|operations|income|revenue|basic|diluted|ronw|nav|per|equity|share|owners|company|segment|attributable)\b|[₹%]", re.I)
_LABEL_NOISE = re.compile(r"^(listed (industry )?peers?\**\s*)", re.I)
_SANE = {"pe": (0.0, 1000.0), "ronw_pct": (-500.0, 200.0), "eps_basic": (-1e4, 1e5), "eps_diluted": (-1e4, 1e5)}
_NOT_COMPANY = re.compile(r"^(source|notes?|particulars|weighted|average|median|mean|\*|total|our company|the company)", re.I)


@dataclass
class PeerRow:
    name: str
    page: int
    values: dict[str, float | None] = field(default_factory=dict)
    revenue_inr: float | None = None
    is_subject: bool = False  # the issuer itself, listed in the same table
    name_reliable: bool = True  # False when a wrapped name could not be rebuilt cleanly


@dataclass
class PeerTable:
    peers: list[PeerRow] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    unit: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def as_peer_companies(self) -> list[dict]:
        """Peers in the shape of `valuation.comparables.PeerCompany` (the issuer's own row excluded)."""
        out = []
        for p in self.peers:
            if p.is_subject:
                continue
            out.append({"name": p.name or f"Peer (see DRHP p.{p.page})", "pe": p.values.get("pe"), "eps": p.values.get("eps_diluted") or p.values.get("eps_basic"),
                        "revenue": p.revenue_inr, "name_verified": p.name_reliable, "source": f"DRHP p.{p.page} (Basis for Offer Price)"})
        return out


def _column_field(header: str) -> str | None:
    h = header.lower()
    return next((f for pat, f in COLUMNS if re.search(pat, h)), None)


def _parse_page(page: "pymupdf.Page", number: int, company_hint: str | None) -> tuple[list[PeerRow], str | None]:
    lines = _lines(page)
    parsed = [(ln,) + _split_line(ln) for ln in lines]
    head_i = next((i for i, (ln, label, _) in enumerate(parsed) if re.search(r"name of the", label, re.I)), None)
    if head_i is None:
        return [], None
    top = parsed[head_i][0][0][1] - 2
    data_idx = [i for i in range(head_i + 1, len(parsed)) if len(parsed[i][2]) >= 3]
    if len(data_idx) < 2:
        return [], None
    first_y = parsed[data_idx[0]][0][0][1]
    clusters = [c for c in _cluster([w[2] for i in data_idx for w in parsed[i][2]], gap=14.0) if len(c) >= 2]
    if len(clusters) < 3:
        return [], None
    heads = []
    words = [w for ln, _, _ in parsed for w in ln if top <= w[1] < first_y - 1 and w[3] <= first_y + 1]
    for k, c in enumerate(clusters):
        left = max(clusters[k - 1]) + 6 if k else min(c) - 90
        heads.append(" ".join(w[4] for w in sorted((w for w in words if left < (w[0] + w[2]) / 2 <= max(c) + 12),
                                                   key=lambda w: (round(w[1] / 3), w[0]))))
    fields = [_column_field(h) for h in heads]
    unit = detect_unit(" ".join(w[4] for w in words)) or detect_unit(page.get_text()[:1500])

    def col(x: float) -> int | None:
        k = min(range(len(clusters)), key=lambda j: max(min(clusters[j]) - x, x - max(clusters[j]), 0.0))
        return k if max(min(clusters[k]) - x, x - max(clusters[k]), 0.0) <= 25 else None

    rows: list[PeerRow] = []
    consumed: set[int] = set()
    prev_numeric = head_i
    for i in data_idx:
        ln, label, vals = parsed[i]
        parts = [parsed[j][1] for j in range(prev_numeric + 1, i) if j not in consumed and parsed[j][1] and not parsed[j][2]]
        # Names wrap over several lines; the figures sit on one of them. A trailing "Limited" follows.
        nxt = parsed[i + 1] if i + 1 < len(parsed) else None
        tail = [nxt[1]] if nxt and not nxt[2] and nxt[1] and _TAIL.match(nxt[1]) else []
        if tail:
            consumed.add(i + 1)
        name = " ".join(parts + [label] + tail).strip()
        prev_numeric = i + (1 if tail else 0)
        if not name or _NOT_COMPANY.match(name) and not company_hint:
            continue
        name = _LABEL_NOISE.sub("", re.sub(r"\s+", " ", name)).strip()
        row = PeerRow(name=name, page=number, is_subject=bool(_LEAK.search(name)),
                      name_reliable=bool(re.search(r"(limited|ltd\.?|corporation|services|finance|energy|industries)\s*(\(\d\))?$", name, re.I))
                      and len(re.findall(r"\b(limited|ltd)\b", name, re.I)) <= 1)
        for w in vals:
            k = col(w[2])
            if k is not None and fields[k]:
                row.values[fields[k]] = parse_number(w[4])
        if unit and row.values.get("revenue") is not None:
            row.revenue_inr = to_inr(row.values["revenue"], unit[0])
        for k, (lo, hi) in _SANE.items():
            if row.values.get(k) is not None and not lo < row.values[k] < hi:
                row.values[k] = None
        if row.values.get("pe") is not None:
            rows.append(row)
    return rows, (unit[0] if unit else None)


def extract_peers(pdf_path: str, pages: list[int], company_name: str | None = None) -> PeerTable:
    """Peer table from the given 1-based pages (the Basis for Offer Price chapter)."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    table = PeerTable()
    with pymupdf.open(pdf_path) as doc:
        text = " ".join(doc[n - 1].get_text() for n in pages)
        if re.search(r"no listed (entities|companies|peers)[^.]{0,120}(comparable|similar)|not (any )?(listed )?(peers|comparable)", text, re.I):
            table.note = "The DRHP states that there are no comparable listed peers."
        for n in pages:
            rows, unit = _parse_page(doc[n - 1], n, company_name)
            if not rows:
                continue
            table.pages.append(n)
            table.unit = table.unit or unit
            table.peers += rows
    for p in table.peers:  # only the first page can start with the issuer's own row; later, a header-leaked name is a lost peer name
        if p.is_subject and p.page != table.pages[0]:
            p.is_subject, p.name_reliable, p.name = False, False, ""
    if company_name:  # a leaked-header name already marks the issuer's own row; this catches a clean one
        key = re.sub(r"\W+", "", company_name.lower())[:8]
        for p in table.peers:
            p.is_subject = p.is_subject or (bool(key) and re.sub(r"\W+", "", p.name.lower()).startswith(key))
    if not table.peers and not table.note:
        table.note = ("No peer comparison table with figures was found (the DRHP may leave it as [●] until the price band "
                      "is fixed). Enter peers manually.")
    return table
