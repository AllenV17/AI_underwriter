"""Section mapping, chunking, fusion, BM25 and citations."""

from app.ingestion.pdf_parser import detect_company_name, detect_doc_type, validate_pdf, InvalidPDFError
from app.ingestion.section_mapper import chapter_key, map_sections
from app.rag.bm25 import BM25Index, tokenize
from app.rag.chunker import Chunk, chunk_document
from app.rag.citations import resolve_citations
from app.rag.fusion import reciprocal_rank_fusion
from app.schemas.document_ir import DocumentIR, PageIR, TableIR
from app.services.retrieval_service import Evidence
import pytest


def test_chapter_keys():
    assert chapter_key("BASIS FOR ISSUE PRICE") == "basis_for_offer_price"
    assert chapter_key("OTHER FINANCIAL INFORMATION") == "other_financial_info"
    assert chapter_key("MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION") == "mdna"


def test_map_sections_sample(sample_pdf):
    import pymupdf

    texts = [p.get_text() for p in pymupdf.open(sample_pdf)]
    chapters, offset, agree, warnings = map_sections(texts)
    assert offset == 1 and agree == 1.0  # printed page 1 is PDF index 2 (0-based)
    by_key = {c.key: c for c in chapters}
    assert by_key["risk_factors"].pdf_start == 4
    assert by_key["restated_financials"].pdf_start == 6 and by_key["restated_financials"].pdf_end == 8
    # The synthetic statement page does not print the chapter heading: the self-check must flag it.
    assert [c.key for c in chapters if not c.title_verified] == ["restated_financials"]
    assert any("not found on their mapped start page" in w for w in warnings)


def test_cover_detection(sample_pdf):
    import pymupdf

    cover = pymupdf.open(sample_pdf)[0].get_text()
    assert detect_doc_type(cover) == "DRHP"
    assert detect_company_name(cover) == "Acme Widgets Limited"


def test_validate_rejects_non_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"hello")
    with pytest.raises(InvalidPDFError):
        validate_pdf(p)


def test_rrf_prefers_items_ranked_by_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [i for i, _ in fused][:2] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def _ir() -> DocumentIR:
    long = " ".join(f"Sentence number {i} about widgets and customers." for i in range(120))
    pages = [PageIR(page_number=1, text="cover", chapter_key="front_matter"),
             PageIR(page_number=2, text=long, chapter="RISK FACTORS", chapter_key="risk_factors"),
             PageIR(page_number=3, text="numbers 1 2 3", chapter="RESTATED", chapter_key="restated_financials")]
    tables = [TableIR(table_id="stmt-pl", page_number=3, page_numbers=[3], chapter_key="restated_financials",
                      title="P&L", header=["Particulars", "FY2025"], rows=[["Revenue", "1,000"]] * 60, engine="statement-parser")]
    return DocumentIR(document_id="d1", company_name="Acme", filename="a.pdf", n_pages=3, pages=pages, tables=tables)


def test_chunker_respects_pages_and_tables():
    chunks = chunk_document(_ir())
    assert all(c.page_number in (2, 3) for c in chunks)
    assert not any(c.page_number == 3 and c.chunk_type == "paragraph" for c in chunks)  # statement page -> table only
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 2 and all("| Particulars | FY2025 |" in c.text for c in tables)  # header repeated per part
    paras = [c for c in chunks if c.chunk_type == "paragraph"]
    assert len(paras) > 1 and all(len(c.text.split()) <= 420 for c in paras)
    assert paras[0].section_label == "Risk Factors"


def test_bm25_exact_terms():
    cs = [Chunk(chunk_id=str(i), document_id="d", page_number=i, text=t) for i, t in
          enumerate(["revenue was 1,234.5 million", "risks of litigation", "promoter holding 75%"])]
    res = BM25Index(cs).search("1234.5", 3)
    assert res and res[0][0].chunk_id == "0"
    assert "1234.5" in tokenize("₹1,234.5")


def test_citations_resolved_and_invalid_dropped():
    c = Chunk(chunk_id="x", document_id="d", page_number=45, chapter_key="risk_factors", text="t")
    ev = [Evidence(ref="S1", chunk=c)]
    out = resolve_citations("Customer concentration is high [S1]. Margins rose [S7].", ev)
    assert "(Source: DRHP, Risk Factors, page 45)" in out.text
    assert "S7" not in out.text and out.invalid_refs == ["S7"]
    assert out.citations[0]["page_number"] == 45


def test_clean_text_removes_surrogates_and_mojibake():
    from app.ingestion.pdf_parser import clean_text

    assert clean_text("bad\udc9dchar â€¢ item") == "badchar • item"


def test_intent_routing():
    from app.agent.orchestrator import classify

    assert classify("How concentrated is revenue among top customers?") == "risk"
    assert classify("What was revenue growth in FY2023?") == "financial_health"
    assert classify("Estimate a fair value for the IPO") == "valuation"
    assert classify("Who are the promoters?") == "management"
    assert classify("What are the objects of the offer?") == "offer"


def test_intent_routing_regressions():
    from app.agent.orchestrator import classify

    for q, intent in [("What does the company apply the net proceeds to?", "offer"),
                      ("What is the net worth of the company?", "financial_health"),
                      ("Which regulatory approvals are pending?", "approvals"),
                      ("How many employees does the company have?", "business"),
                      ("What are the key customers?", "business"),
                      ("Who sits on the board of directors?", "management"),
                      ("Where is the registrar and what is the lot size?", "offer")]:
        assert classify(q) == intent, q
