"""Map every DRHP page to its section and chapter using the printed table of contents.

SEBI's ICDR Regulations fix the order of an offer document's sections, and every DRHP
prints a table of contents in its first pages (`SECTION II: RISK FACTORS ........ 27`).
PDF bookmarks are usually missing, so the printed contents page is the reliable source.
This is the Stage 2 parser (358/358 chapters located on the 10 pilot DRHPs), packaged
for the application.
"""

import re
from collections import Counter

from app.schemas.document_ir import Chapter

TOC_ENTRY = re.compile(r"([A-Z][^.]*?[A-Z\)])\s*\.{2,}\s*(\d{1,4})\b")
SECTION_HEAD = re.compile(r"^SECTION\s+([IVXL]+)\s*[:\-–]\s*(.+)$")
PAGE_NUM_LINE = re.compile(r"^\d{1,4}$")

# Canonical chapter keys. Order matters: specific patterns before generic ones.
CHAPTER_KEYS: list[tuple[str, str]] = [
    ("definitions", r"DEFINITIONS AND ABBREVIATIONS"),
    ("offer_document_summary", r"(OFFER DOCUMENT SUMMARY|SUMMARY OF (THE |THIS )?(OFFER DOCUMENT|ISSUE DOCUMENT|DRAFT RED HERRING|RED HERRING|PROSPECTUS))"),
    ("presentation_conventions", r"CERTAIN CONVENTIONS"),
    ("forward_looking", r"FORWARD[- ]LOOKING"),
    ("risk_factors", r"RISK FACTORS"),
    ("the_offer", r"^THE (OFFER|ISSUE)$"),
    ("financial_summary", r"SUMMARY (OF )?(RESTATED )?(CONSOLIDATED )?FINANCIAL"),
    ("general_information", r"GENERAL INFORMATION"),
    ("capital_structure", r"CAPITAL STRUCTURE"),
    ("objects_of_offer", r"OBJECTS? OF THE (OFFER|ISSUE)"),
    ("basis_for_offer_price", r"BASIS (FOR|OF) (THE )?(OFFER|ISSUE) PRICE"),
    ("tax_benefits", r"TAX BENEFITS"),
    ("industry_overview", r"INDUSTRY OVERVIEW"),
    ("our_business", r"^OUR BUSINESS"),
    ("regulations", r"KEY (INDUSTRY )?REGULATIONS"),
    ("history_corporate", r"HISTORY AND (CERTAIN )?CORPORATE"),
    ("subsidiaries", r"(OUR )?SUBSIDIAR"),
    ("management", r"^OUR MANAGEMENT"),
    ("promoters", r"PROMOTER"),
    ("group_companies", r"GROUP COMPAN"),
    ("dividend_policy", r"DIVIDEND POLICY"),
    ("other_financial_info", r"OTHER FINANCIAL INFORMATION"),
    ("statistical_info", r"SELECTED STATISTICAL INFORMATION"),
    ("restated_financials", r"(RESTATED|FINANCIAL STATEMENTS|FINANCIAL INFORMATION$)"),
    ("related_party", r"RELATED PARTY"),
    ("mdna", r"MANAGEMENT.S DISCUSSION"),
    ("capitalisation", r"CAPITALI[SZ]ATION STATEMENT"),
    ("indebtedness", r"(FINANCIAL )?INDEBTEDNESS|BORROWINGS"),
    ("litigation", r"LITIGATION"),
    ("approvals", r"APPROVALS"),
    ("regulatory_disclosures", r"OTHER REGULATORY AND STATUTORY"),
    ("offer_terms", r"TERMS OF THE (OFFER|ISSUE)"),
    ("offer_structure", r"(OFFER|ISSUE) STRUCTURE|STRUCTURE OF THE (OFFER|ISSUE)"),
    ("offer_procedure", r"(OFFER|ISSUE) PROCEDURE|PROCEDURE OF THE (OFFER|ISSUE)"),
    ("foreign_ownership", r"FOREIGN OWNERSHIP"),
    ("articles", r"ARTICLES OF ASSOCIATION"),
    ("material_contracts", r"MATERIAL CONTRACTS"),
    ("declaration", r"DECLARATION"),
]

# Human-readable names for the sections the application surfaces.
KEY_LABELS = {
    "offer_document_summary": "Summary", "risk_factors": "Risk Factors", "industry_overview": "Industry Overview",
    "our_business": "Business Overview", "restated_financials": "Financial Information", "mdna": "MD&A",
    "management": "Management", "promoters": "Promoters", "capital_structure": "Capital Structure",
    "objects_of_offer": "Objects of the Issue", "basis_for_offer_price": "Basis for Offer Price",
    "litigation": "Outstanding Litigation", "related_party": "Related Party Transactions",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chapter_key(title: str) -> str:
    """Canonical key for a chapter title, or 'other'."""
    for key, pat in CHAPTER_KEYS:
        if re.search(pat, title.upper()):
            return key
    return "other"


def find_toc_pages(page_texts: list[str], search_pages: int = 15) -> list[int]:
    """0-based indices of the table-of-contents page(s)."""
    start = next((i for i, t in enumerate(page_texts[:search_pages]) if "TABLE OF CONTENTS" in t.upper()), None)
    if start is None:
        # Some documents title it just "CONTENTS"; require several dotted entries too.
        start = next((i for i, t in enumerate(page_texts[:search_pages])
                      if "CONTENTS" in t[:400].upper() and len(TOC_ENTRY.findall(norm(t))) >= 5), None)
    if start is None:
        return []
    pages = [start]
    for i in range(start + 1, min(start + 4, len(page_texts))):
        if len(TOC_ENTRY.findall(norm(page_texts[i]))) >= 3:
            pages.append(i)
        else:
            break
    return pages


def parse_toc(page_texts: list[str], toc_pages: list[int]) -> list[tuple[str, int]]:
    """(title, printed page) entries from the contents pages."""
    text = " ".join(norm(page_texts[i]) for i in toc_pages)
    text = re.sub(r"(?i)table of contents", " ", text)
    out = []
    for m in TOC_ENTRY.finditer(text):
        title = norm(m.group(1)).strip(" .").replace("�", "'").replace("’", "'")
        out.append((title, int(m.group(2))))
    return out


def printed_number(text: str) -> int | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:2] + lines[-2:]:
        if PAGE_NUM_LINE.match(ln):
            return int(ln)
    return None


def page_offset(page_texts: list[str]) -> tuple[int | None, float]:
    """Most common (pdf_index - printed_number) and the share of numbered pages agreeing."""
    diffs = [i - n for i, t in enumerate(page_texts) if (n := printed_number(t)) is not None]
    if not diffs:
        return None, 0.0
    off, cnt = Counter(diffs).most_common(1)[0]
    return off, cnt / len(diffs)


def build_chapters(entries: list[tuple[str, int]], offset: int, n_pages: int, page_texts: list[str]) -> list[Chapter]:
    """Chapters with 1-based PDF page ranges; section headers collapse into their first chapter."""
    raw: list[dict] = []
    section = None
    for title, printed in entries:
        m = SECTION_HEAD.match(title)
        if m:
            section = f"SECTION {m.group(1)}: {m.group(2)}"
            title = m.group(2)
        start = printed + offset  # 0-based index
        if not 0 <= start < n_pages:
            continue
        item = {"section": section, "title": title, "key": chapter_key(title), "printed_start": printed, "start": start}
        if raw and raw[-1]["start"] == start:
            raw[-1] = item
        else:
            raw.append(item)
    raw.sort(key=lambda c: c["start"])
    chapters = []
    for i, c in enumerate(raw):
        end = (raw[i + 1]["start"] - 1) if i + 1 < len(raw) else n_pages - 1
        chapters.append(Chapter(
            section=c["section"], title=c["title"], key=c["key"], printed_start=c["printed_start"],
            pdf_start=c["start"] + 1, pdf_end=max(end, c["start"]) + 1,
            title_verified=_title_on_page(c["title"], page_texts[c["start"]]),
        ))
    return chapters


def _title_on_page(title: str, text: str) -> bool:
    head = norm(text[:1500]).upper()
    words = re.findall(r"[A-Z]{3,}", title.upper())[:4]
    return bool(words) and all(w in head for w in words)


def map_sections(page_texts: list[str]) -> tuple[list[Chapter], int | None, float, list[str]]:
    """Return (chapters, page_offset, offset_agreement, warnings) for a document."""
    warnings: list[str] = []
    toc = find_toc_pages(page_texts)
    if not toc:
        warnings.append("No table of contents found in the first 15 pages; sections could not be mapped.")
        return [], None, 0.0, warnings
    entries = parse_toc(page_texts, toc)
    offset, agree = page_offset(page_texts)
    if offset is None:
        warnings.append("No printed page numbers found; assuming printed page 1 follows the contents page.")
        offset = toc[-1] + 1 - 1
    if agree < 0.8:
        warnings.append(f"Printed page numbers agree with the computed offset on only {agree:.0%} of pages.")
    chapters = build_chapters(entries, offset, len(page_texts), page_texts)
    unverified = [c.title for c in chapters if not c.title_verified]
    if unverified:
        warnings.append(f"{len(unverified)} chapter title(s) not found on their mapped start page: {unverified[:5]}")
    return chapters, offset, agree, warnings


def assign_pages(n_pages: int, chapters: list[Chapter]) -> list[tuple[str | None, str | None, str]]:
    """(section, chapter title, chapter key) for each 0-based page index."""
    out: list[tuple[str | None, str | None, str]] = [(None, "Front matter", "front_matter")] * n_pages
    for c in chapters:
        for p in range(c.pdf_start - 1, c.pdf_end):
            out[p] = (c.section, c.title, c.key)
    return out
