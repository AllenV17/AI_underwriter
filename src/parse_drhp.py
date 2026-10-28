"""Parse DRHP/RHP PDFs into page-level text tagged with SEBI ICDR sections.

For each PDF in the input directory this writes data/processed/<doc_id>/:
  pages.jsonl    one record per PDF page: page numbers, section, chapter, text
  sections.json  the chapter map recovered from the document's table of contents

Usage:
  python src/parse_drhp.py                      # all PDFs in data/raw/drhp
  python src/parse_drhp.py --only Bajaj         # files whose name contains "Bajaj"
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "raw" / "drhp"
DEFAULT_OUT = ROOT / "data" / "processed"

# TOC line: "TITLE ........ 123" (titles may wrap onto the next line before the dots).
TOC_ENTRY = re.compile(r"([A-Z][^.]*?[A-Z\)])\s*\.{2,}\s*(\d{1,4})\b")
SECTION_HEAD = re.compile(r"^SECTION\s+([IVXL]+)\s*[:\-–]\s*(.+)$")
PAGE_NUM_LINE = re.compile(r"^\d{1,4}$")

# Canonical keys for the chapters downstream stages care about. Titles vary between
# documents ("BASIS FOR OFFER PRICE" vs "BASIS FOR ISSUE PRICE"), so match by pattern.
CHAPTER_KEYS = [
    ("definitions", r"DEFINITIONS AND ABBREVIATIONS"),
    ("offer_document_summary", r"(OFFER DOCUMENT SUMMARY|SUMMARY OF (THE |THIS )?(OFFER DOCUMENT|ISSUE DOCUMENT|DRAFT RED HERRING|RED HERRING))"),
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


def chapter_key(title: str) -> str:
    for key, pat in CHAPTER_KEYS:
        if re.search(pat, title):
            return key
    return "other"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_toc_pages(doc) -> list[int]:
    """Indices of the table-of-contents page(s) within the first 15 pages."""
    start = None
    for i in range(min(15, len(doc))):
        if "TABLE OF CONTENTS" in doc[i].get_text().upper():
            start = i
            break
    if start is None:
        return []
    pages = [start]
    # TOC can run onto following pages; continue while pages are mostly dotted entries.
    for i in range(start + 1, min(start + 4, len(doc))):
        if len(TOC_ENTRY.findall(norm(doc[i].get_text()))) >= 3:
            pages.append(i)
        else:
            break
    return pages


def parse_toc(doc, toc_pages: list[int]) -> list[dict]:
    text = " ".join(norm(doc[i].get_text()) for i in toc_pages)
    text = text.replace("TABLE OF CONTENTS", " ")
    entries = []
    for m in TOC_ENTRY.finditer(text):
        # Apostrophes often extract as U+FFFD or curly quotes; normalise to ASCII.
        title = norm(m.group(1)).strip(" .").replace("�", "'").replace("’", "'")
        entries.append({"title": title, "printed_start": int(m.group(2))})
    return entries


def printed_number(page) -> int | None:
    """Page number printed at the top or bottom of the page, if any."""
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    for l in lines[:2] + lines[-2:]:
        if PAGE_NUM_LINE.match(l):
            return int(l)
    return None


def page_offset(doc) -> tuple[int | None, float]:
    """Most common (pdf_index - printed_number); returns (offset, share of pages agreeing)."""
    diffs = []
    for i in range(len(doc)):
        n = printed_number(doc[i])
        if n is not None:
            diffs.append(i - n)
    if not diffs:
        return None, 0.0
    off, cnt = Counter(diffs).most_common(1)[0]
    return off, cnt / len(diffs)


def build_chapters(entries: list[dict], offset: int, n_pages: int) -> list[dict]:
    chapters, section = [], None
    for e in entries:
        m = SECTION_HEAD.match(e["title"])
        if m:
            section = f"SECTION {m.group(1)}: {m.group(2)}"
            # A section with no separate chapter under it (e.g. RISK FACTORS) is its own chapter.
            title = m.group(2)
        else:
            title = e["title"]
        start = e["printed_start"] + offset
        chapters.append({
            "section": section,
            "title": title,
            "key": chapter_key(title),
            "printed_start": e["printed_start"],
            "pdf_start": start,
        })
    # Drop a section header immediately followed by a chapter on the same page.
    dedup = []
    for c in chapters:
        if dedup and dedup[-1]["pdf_start"] == c["pdf_start"]:
            dedup[-1] = c
        else:
            dedup.append(c)
    for a, b in zip(dedup, dedup[1:]):
        a["pdf_end"] = b["pdf_start"] - 1
    if dedup:
        dedup[-1]["pdf_end"] = n_pages - 1
    return dedup


def title_found(doc, ch: dict) -> bool:
    """Does the chapter title appear near the top of its mapped start page?"""
    if not 0 <= ch["pdf_start"] < len(doc):
        return False
    head = norm(doc[ch["pdf_start"]].get_text()[:1500]).upper()
    words = [w for w in re.findall(r"[A-Z]{3,}", ch["title"])][:4]
    return all(w in head for w in words)


def parse_pdf(path: Path, out_root: Path) -> dict:
    doc = pymupdf.open(path)
    doc_id = path.stem
    toc_pages = find_toc_pages(doc)
    entries = parse_toc(doc, toc_pages) if toc_pages else []
    offset, agree = page_offset(doc)
    chapters = build_chapters(entries, offset or 0, len(doc)) if entries else []

    out_dir = out_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    page_ch = {}
    for c in chapters:
        for p in range(c["pdf_start"], c["pdf_end"] + 1):
            page_ch[p] = c

    low_text = 0
    with open(out_dir / "pages.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(doc)):
            text = doc[i].get_text("text", sort=True)
            c = page_ch.get(i)
            rec = {
                "doc_id": doc_id,
                "pdf_page": i + 1,
                "printed_page": (i - offset) if offset is not None and i - offset >= 1 else None,
                "section": c["section"] if c else None,
                "chapter": c["title"] if c else ("FRONT MATTER" if i < (chapters[0]["pdf_start"] if chapters else 0) else None),
                "chapter_key": c["key"] if c else "front_matter",
                "n_chars": len(text),
                "text": text,
            }
            if len(text.strip()) < 200:
                low_text += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    found = [title_found(doc, c) for c in chapters]
    for c, ok in zip(chapters, found):
        c["title_verified"] = ok
        c["pdf_start"] += 1  # store 1-based for humans
        c["pdf_end"] += 1
    (out_dir / "sections.json").write_text(json.dumps({
        "doc_id": doc_id,
        "n_pages": len(doc),
        "toc_pdf_pages": [p + 1 for p in toc_pages],
        "page_offset": offset,
        "page_offset_agreement": round(agree, 3),
        "chapters": chapters,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    keys = {c["key"] for c in chapters}
    return {
        "doc_id": doc_id,
        "pages": len(doc),
        "chapters": len(chapters),
        "verified": f"{sum(found)}/{len(found)}",
        "offset": offset,
        "offset_agree": round(agree, 2),
        "low_text_pages": low_text,
        "missing_key": [k for k in ("risk_factors", "objects_of_offer", "basis_for_offer_price",
                                    "our_business", "restated_financials", "mdna") if k not in keys],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", help="substring filter on file name")
    args = ap.parse_args()

    pymupdf.TOOLS.mupdf_display_errors(False)
    files = sorted(p for p in args.in_dir.glob("*") if p.suffix.lower() == ".pdf")
    if args.only:
        files = [p for p in files if args.only.lower() in p.name.lower()]
    if not files:
        sys.exit(f"No PDFs found in {args.in_dir}")

    print(f"{'doc':28} {'pages':>5} {'chap':>4} {'verified':>8} {'offset':>6} {'agree':>5} {'lowtxt':>6}  missing")
    for p in files:
        r = parse_pdf(p, args.out_dir)
        print(f"{r['doc_id']:28} {r['pages']:5} {r['chapters']:4} {r['verified']:>8} {str(r['offset']):>6} "
              f"{r['offset_agree']:5} {r['low_text_pages']:6}  {','.join(r['missing_key']) or '-'}")


if __name__ == "__main__":
    main()
