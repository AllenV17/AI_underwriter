"""DRHP ingestion pipeline.

    validate -> extract pages (+OCR) -> map sections -> detect company -> financial
    statements -> chapter tables -> save IR -> chunk -> embed + index (Qdrant) -> ready

Each stage updates the document's status row so the frontend can show progress. A
failure in indexing does not discard the parsed document: it is marked ready with
`indexed=False` and a warning, and can be re-indexed.
"""

import logging
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.ingestion.pdf_parser import detect_company_name, detect_doc_type, extract_pages, validate_pdf
from app.ingestion.section_mapper import assign_pages, map_sections
from app.ingestion.table_extractor import extract_chapter_tables, statement_tables
from app.models.db import Document
from app.rag.chunker import Chunk, chunk_document
from app.rag.vector_store import VectorStore
from app.schemas.document_ir import DocumentIR, PageIR
from app.services.embedding_service import get_embedder
from app.services.financial_extraction_service import extract_and_save_peers, extract_financials, persist_financials
from app.utils.storage import chunks_path, save_ir, upload_path

log = logging.getLogger(__name__)


def create_document(session: Session, filename: str, src: Path | bytes) -> Document:
    """Store an uploaded PDF and create its database row (status 'uploaded')."""
    doc_id = str(uuid.uuid4())
    dest = upload_path(doc_id)
    if isinstance(src, bytes):
        dest.write_bytes(src)
    else:
        shutil.copyfile(src, dest)
    try:
        n_pages = validate_pdf(dest, get_settings().max_upload_mb)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    doc = Document(id=doc_id, filename=filename, n_pages=n_pages, status="uploaded", stage="uploaded", progress=0.0)
    session.add(doc)
    session.commit()
    return doc


class _Progress:
    def __init__(self, factory: sessionmaker, document_id: str) -> None:
        self.factory, self.document_id = factory, document_id

    def __call__(self, stage: str, progress: float, **fields) -> None:
        with self.factory() as s:
            doc = s.get(Document, self.document_id)
            if doc is None:
                return
            doc.stage, doc.progress = stage, round(progress, 3)
            for k, v in fields.items():
                setattr(doc, k, v)
            s.commit()
        log.info("[%s] %s (%.0f%%)", self.document_id[:8], stage, progress * 100)


def build_ir(document_id: str, filename: str, pdf_path: Path, progress=None) -> DocumentIR:
    """Pages, section map, company metadata. No financial or RAG work."""
    s = get_settings()
    progress and progress("extracting text", 0.05)
    raw = extract_pages(pdf_path, ocr_enabled=s.ocr_enabled)
    texts = [p.text for p in raw.pages]
    progress and progress("mapping sections", 0.25)
    chapters, offset, agree, warnings = map_sections(texts)
    assigned = assign_pages(len(texts), chapters)
    pages: list[PageIR] = []
    current_chapter, sub = None, None
    for rp, (section, chapter, key) in zip(raw.pages, assigned):
        if chapter != current_chapter:
            current_chapter, sub = chapter, None
        heads = [h for h in rp.headings if not chapter or h.upper() != chapter.upper()]
        if heads:
            sub = heads[0]
        pages.append(PageIR(page_number=rp.number, printed_page=(rp.number - 1 - offset) if offset is not None and rp.number - 1 - offset >= 1 else None,
                            section=section, chapter=chapter, chapter_key=key, subsection=sub, text=rp.text, ocr=rp.ocr))
    cover = "\n".join(texts[:2])
    return DocumentIR(
        document_id=document_id, company_name=detect_company_name(cover), filename=filename, n_pages=len(pages),
        doc_type=detect_doc_type(texts[0] if texts else ""), page_offset=offset, page_offset_agreement=agree,
        chapters=chapters, pages=pages, warnings=raw.warnings + warnings,
    )


def index_chunks(document_id: str, chunks: list[Chunk], progress=None, store: VectorStore | None = None) -> None:
    store = store or VectorStore()
    embedder = get_embedder()
    store.delete_document(document_id)
    step = 64
    for i in range(0, len(chunks), step):
        part = chunks[i:i + step]
        vecs = embedder.embed_documents([c.embedding_text() for c in part])
        store.upsert(part, vecs)
        if progress:
            progress(f"embedding chunks ({min(i + step, len(chunks))}/{len(chunks)})", 0.6 + 0.38 * (i + step) / max(len(chunks), 1))


def process_document(document_id: str, factory: sessionmaker) -> None:
    """Run the full pipeline. Safe to call from a background task."""
    s = get_settings()
    progress = _Progress(factory, document_id)
    with factory() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"Unknown document {document_id}")
        filename = doc.filename
    pdf = upload_path(document_id)
    progress("validating", 0.01, status="processing", error=None)
    try:
        validate_pdf(pdf, s.max_upload_mb)
        ir = build_ir(document_id, filename, pdf, progress)

        progress("extracting financial statements", 0.35, company_name=ir.company_name, doc_type=ir.doc_type,
                 n_pages=ir.n_pages)
        data, tables = extract_financials(ir, pdf)
        ir.tables = statement_tables(tables)
        progress("extracting tables", 0.45)
        ir.tables += extract_chapter_tables(pdf, ir.pages, s.table_engine)
        save_ir(ir)
        try:
            extract_and_save_peers(ir, pdf)
        except Exception:  # peers are a convenience; never fail processing over them
            log.exception("Peer table extraction failed for %s", document_id)
        with factory() as session:
            persist_financials(session, document_id, data)
        progress("chunking", 0.55, financials_extracted=bool(data.values), warnings=ir.warnings + data.warnings)

        chunks = chunk_document(ir)
        with open(chunks_path(document_id), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        try:
            index_chunks(document_id, chunks, progress)
            progress("ready", 1.0, status="ready", indexed=True)
        except Exception as e:
            log.exception("Indexing failed for %s", document_id)
            progress("ready (lexical search only)", 1.0, status="ready", indexed=False,
                     warnings=ir.warnings + data.warnings + [f"Vector indexing failed: {e}. BM25 search still works."])
    except Exception as e:
        log.exception("Processing failed for %s", document_id)
        progress("failed", 1.0, status="failed", error=str(e)[:2000])
