"""Measure extraction accuracy and (lexical) retrieval against hand-checked data.

Usage (repo root):
  python src/eval_pipeline.py extraction     # data/eval/extraction_answers.csv vs the extractor
  python src/eval_pipeline.py retrieval      # data/eval/retrieval_questions.csv, BM25 hit@k by chapter

extraction: each answer was read by hand from the cited page of the PDF. A row counts as
correct when the value matches to 0.01 in the printed unit AND the source page matches.
retrieval: ground truth is the chapter a question belongs to (from the verified table-of-contents
map); a hit means a top-k chunk comes from that chapter. Only BM25 is evaluated here; dense
retrieval and the reranker need the embedding models and are not included.
"""

import sys
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "underwriter-ai" / "backend"))

from app.financial_engine.extractor import map_fields  # noqa: E402
from app.financial_engine.statement_parser import locate_and_parse  # noqa: E402
from app.ingestion.section_mapper import map_sections  # noqa: E402

PDFS = ROOT / "data" / "raw" / "drhp"


def extraction() -> None:
    ans = pd.read_csv(ROOT / "data" / "eval" / "extraction_answers.csv")
    out = []
    for f, grp in ans.groupby("file"):
        texts = [p.get_text() for p in pymupdf.open(PDFS / f)]
        chapters, *_ = map_sections(texts)
        fin = sorted([c for c in chapters if c.key == "restated_financials"], key=lambda c: c.pdf_start - c.pdf_end)
        data = map_fields(locate_and_parse(str(PDFS / f), [n for c in fin for n in range(c.pdf_start, c.pdf_end + 1)], len(texts)))
        for _, r in grp.iterrows():
            v = data.get(r["field"], r["period"])
            ok_val = v is not None and v.value_original is not None and abs(v.value_original - r["value"]) <= 0.01
            ok_page = v is not None and v.source_page == r["page"]
            out.append({**r.to_dict(), "extracted": None if v is None else v.value_original, "value_ok": ok_val, "page_ok": ok_page})
    df = pd.DataFrame(out)
    print(df.drop(columns=["read_from"]).to_string(index=False))
    print(f"\nvalue accuracy {df.value_ok.mean():.1%} ({df.value_ok.sum()}/{len(df)}); "
          f"value+page accuracy {(df.value_ok & df.page_ok).mean():.1%}")
    print(df.groupby("file")[["value_ok", "page_ok"]].mean().round(3).to_string())


def retrieval(k_values=(1, 3, 5)) -> None:
    from app.ingestion.pdf_parser import extract_pages
    from app.rag.bm25 import BM25Index
    from app.rag.chunker import chunk_document
    from app.services.document_service import build_ir

    qs = pd.read_csv(ROOT / "data" / "eval" / "retrieval_questions.csv")
    rows = []
    for pdf in sorted(PDFS.glob("*.pdf")):
        ir = build_ir(pdf.stem, pdf.name, pdf)
        index = BM25Index(chunk_document(ir))
        present = {c.key for c in ir.chapters}
        for _, q in qs.iterrows():
            if q.expected_chapter_key not in present:
                continue
            hits = [c.chapter_key for c, _ in index.search(q.question, max(k_values))]
            rows.append({"doc": pdf.stem, "q": q.expected_chapter_key,
                         **{f"hit@{k}": q.expected_chapter_key in hits[:k] for k in k_values},
                         "mrr": next((1 / (i + 1) for i, h in enumerate(hits) if h == q.expected_chapter_key), 0.0)})
    df = pd.DataFrame(rows)
    print(df.groupby("q")[[f"hit@{k}" for k in k_values] + ["mrr"]].mean().round(2).to_string())
    print(f"\n{len(df)} (document, question) pairs; " + "; ".join(f"hit@{k} {df[f'hit@{k}'].mean():.1%}" for k in k_values)
          + f"; MRR {df.mrr.mean():.2f}")


if __name__ == "__main__":
    {"extraction": extraction, "retrieval": retrieval}[sys.argv[1] if len(sys.argv) > 1 else "extraction"]()
