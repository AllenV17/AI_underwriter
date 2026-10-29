"""Join data/manifest.csv prices with features extracted from each DRHP -> data/ipo_history.csv.

Usage (from the repo root):  python src/build_ipo_features.py [--manifest data/manifest.csv] [--out data/ipo_history.csv]

Financial features come from the app's extraction and ratio engine, for the latest full
financial year in the document (all pre-listing, so no leakage). Columns that need manual
reading of the offer chapters (issue_size_inr, fresh_issue_share, nifty_return_30d) are
left blank when not already present in an existing output file, and are kept if you fill
them in. Rows without an issue price and a listing price are written but ignored by training.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "underwriter-ai" / "backend"))

from app.financial_engine.extractor import map_fields  # noqa: E402
from app.financial_engine.peer_extractor import extract_peers  # noqa: E402
from app.financial_engine.statement_parser import locate_and_parse  # noqa: E402
from app.ingestion.section_mapper import map_sections  # noqa: E402
from app.valuation.listing_features import features_from_financials, median_peer_pe  # noqa: E402

MANUAL = ["issue_size_inr", "fresh_issue_share", "nifty_return_30d"]


def features_for(pdf: Path, company: str) -> dict:
    texts = [p.get_text() for p in pymupdf.open(pdf)]
    chapters, *_ = map_sections(texts)
    fin = sorted([c for c in chapters if c.key == "restated_financials"], key=lambda c: c.pdf_start - c.pdf_end)
    pages = [n for c in fin for n in range(c.pdf_start, c.pdf_end + 1)]
    out = features_from_financials(map_fields(locate_and_parse(str(pdf), pages, len(texts))))
    basis = [n for c in chapters if c.key == "basis_for_offer_price" for n in range(c.pdf_start, c.pdf_end + 1)]
    # The issuer's name is passed so its own row is excluded, as in the application.
    out["peer_median_pe"] = median_peer_pe(extract_peers(str(pdf), basis, company).as_peer_companies())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ipo_history.csv"))
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    old = pd.read_csv(a.out).set_index("company") if Path(a.out).exists() else None
    rows = []
    for _, r in man.iterrows():
        pdf = ROOT / "data" / r["file"]
        row = {"company": r["company"], "sector": r["sector"], "listing_date": r["listing_date"], "issue_price": r["issue_price"],
               "listing_price": r["listing_open"], "subscription_total_x": r["subscription_x"]}
        row.update(features_for(pdf, r["company"]) if pdf.exists() else {})
        for m in MANUAL:
            row[m] = old.at[r["company"], m] if old is not None and r["company"] in old.index and m in old.columns else None
        rows.append(row)
        print("built", r["company"], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
