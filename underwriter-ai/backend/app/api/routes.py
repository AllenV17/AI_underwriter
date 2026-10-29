"""REST endpoints."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.agent.orchestrator import UnderwriterAgent
from app.core.compliance import DISCLAIMER, VALUATION_NOTICE
from app.core.config import get_settings
from app.ingestion.pdf_parser import InvalidPDFError
from app.models.db import ChatMessage, Document, Report, get_session, session_factory
from app.schemas.api import (ChatRequest, ChatResponse, ComparableRequest, DCFRequest, DocumentOut, FinancialsResponse,
                             HealthResponse, RatiosResponse, ReportRequest, ReportResponse, SectionOut)
from app.services.document_service import create_document, process_document
from app.services.financial_extraction_service import load_financials, load_peers
from app.services.ratio_service import FinancialsNotAvailableError, compute_ratios
from app.services.report_service import ReportService, render_pdf
from app.services.valuation_service import comparable_valuation, dcf_defaults, dcf_valuation
from app.utils.storage import load_ir

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _doc(session: Session, document_id: str, ready: bool = False) -> Document:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    if ready and doc.status != "ready":
        raise HTTPException(409, f"Document is not ready (status: {doc.status}, stage: {doc.stage})")
    return doc


@router.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    s = get_settings()
    try:
        session.execute(text("SELECT 1"))
        db = "ok"
    except Exception as e:  # report, don't fail the health check
        db = f"error: {type(e).__name__}"
    return HealthResponse(status="ok", llm_configured=s.llm_configured, model=s.openrouter_model,
                          database=db, vector_store=s.qdrant_url if s.qdrant_url.startswith("http") else "embedded",
                          embedding_model=s.embedding_model,
                          reranker=s.reranker_model if s.reranker_enabled else "disabled")


@router.get("/compliance/disclaimer")
def disclaimer() -> dict:
    return {"disclaimer": DISCLAIMER, "valuation_notice": VALUATION_NOTICE}


@router.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload(background: BackgroundTasks, file: UploadFile = File(...), auto_process: bool = True,
                 session: Session = Depends(get_session)) -> Document:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    data = await file.read()
    if len(data) > get_settings().max_upload_mb * 1_048_576:
        raise HTTPException(413, f"File exceeds {get_settings().max_upload_mb} MB.")
    try:
        doc = create_document(session, file.filename, data)
    except InvalidPDFError as e:
        raise HTTPException(400, str(e)) from e
    if auto_process:
        doc.status, doc.stage = "processing", "queued"
        session.commit()
        background.add_task(process_document, doc.id, session_factory())
    return doc


@router.post("/documents/{document_id}/process", response_model=DocumentOut, status_code=202)
def process(document_id: str, background: BackgroundTasks, session: Session = Depends(get_session)) -> Document:
    doc = _doc(session, document_id)
    if doc.status == "processing" and doc.stage not in ("queued",):
        raise HTTPException(409, "Document is already being processed.")
    doc.status, doc.stage, doc.progress = "processing", "queued", 0.0
    session.commit()
    background.add_task(process_document, doc.id, session_factory())
    return doc


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)) -> list[Document]:
    return list(session.scalars(select(Document).order_by(Document.created_at.desc())))


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, session: Session = Depends(get_session)) -> Document:
    return _doc(session, document_id)


@router.get("/documents/{document_id}/sections", response_model=list[SectionOut])
def sections(document_id: str, session: Session = Depends(get_session)) -> list[SectionOut]:
    _doc(session, document_id, ready=True)
    ir = load_ir(document_id)
    if ir is None:
        raise HTTPException(404, "Processed document not found")
    return [SectionOut(title=c.title, key=c.key, section=c.section, pdf_start=c.pdf_start, pdf_end=c.pdf_end,
                       printed_start=c.printed_start, title_verified=c.title_verified) for c in ir.chapters]


@router.post("/documents/{document_id}/chat", response_model=ChatResponse)
def chat(document_id: str, req: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    _doc(session, document_id, ready=True)
    ans = UnderwriterAgent(document_id).run(req.question, shares=req.shares_outstanding, price_per_share=req.price_per_share)
    session.add(ChatMessage(document_id=document_id, role="user", content=req.question))
    session.add(ChatMessage(document_id=document_id, role="assistant", content=ans.answer, citations=ans.citations))
    session.commit()
    return ChatResponse(answer=ans.answer, intent=ans.intent, steps=ans.steps, citations=ans.citations, evidence=ans.evidence,
                        llm_used=ans.llm_used, compliance_flagged=ans.compliance_flagged, warnings=ans.warnings,
                        disclaimer=VALUATION_NOTICE if ans.intent in ("valuation", "listing") else None)


@router.get("/documents/{document_id}/financials", response_model=FinancialsResponse)
def financials(document_id: str, session: Session = Depends(get_session)) -> FinancialsResponse:
    _doc(session, document_id, ready=True)
    data = load_financials(document_id)
    if data is None:
        raise HTTPException(404, "No financial data extracted for this document.")
    return FinancialsResponse(**data.to_dict())


@router.get("/documents/{document_id}/ratios", response_model=RatiosResponse)
def ratios(document_id: str, price_per_share: float | None = None, shares_outstanding: float | None = None,
           session: Session = Depends(get_session)) -> RatiosResponse:
    _doc(session, document_id, ready=True)
    try:
        res, meta = compute_ratios(document_id, price_per_share, shares_outstanding)
    except FinancialsNotAvailableError as e:
        raise HTTPException(404, str(e)) from e
    return RatiosResponse(ratios=[r.to_dict() for r in res], meta=meta)


@router.get("/documents/{document_id}/peers")
def peers(document_id: str, session: Session = Depends(get_session)) -> dict:
    """Peer table as printed in the DRHP (a suggestion to verify against the cited page, not an input)."""
    _doc(session, document_id, ready=True)
    data = load_peers(document_id)
    if data is None:
        raise HTTPException(404, "Peer table not extracted for this document (re-process it).")
    return data


@router.post("/documents/{document_id}/valuation/comparable")
def valuation_comparable(document_id: str, req: ComparableRequest, session: Session = Depends(get_session)) -> dict:
    _doc(session, document_id, ready=True)
    try:
        return comparable_valuation(document_id, req.peers, req.period, req.shares_outstanding, req.overrides)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/documents/{document_id}/valuation/dcf/defaults")
def valuation_dcf_defaults(document_id: str, session: Session = Depends(get_session)) -> dict:
    _doc(session, document_id, ready=True)
    try:
        return dcf_defaults(document_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/documents/{document_id}/valuation/dcf")
def valuation_dcf(document_id: str, req: DCFRequest, session: Session = Depends(get_session)) -> dict:
    _doc(session, document_id, ready=True)
    return dcf_valuation(req.assumptions)


@router.post("/documents/{document_id}/report", response_model=ReportResponse)
def report(document_id: str, req: ReportRequest, session: Session = Depends(get_session)) -> ReportResponse:
    _doc(session, document_id, ready=True)
    content = ReportService(document_id).build(
        peers=req.peers, shares=req.shares_outstanding, issue_price=req.issue_price,
        dcf_assumptions=req.dcf_assumptions.model_dump() if req.dcf_assumptions else None)
    pdf = render_pdf(content)
    row = Report(document_id=document_id, content=content, pdf_path=str(pdf))
    session.add(row)
    session.commit()
    return ReportResponse(report_id=row.id, report=content, pdf_url=f"/api/documents/{document_id}/report/{row.id}/pdf")


@router.get("/documents/{document_id}/report/{report_id}/pdf")
def report_pdf(document_id: str, report_id: str, session: Session = Depends(get_session)) -> FileResponse:
    row = session.get(Report, report_id)
    if row is None or row.document_id != document_id or not row.pdf_path:
        raise HTTPException(404, "Report not found")
    return FileResponse(row.pdf_path, media_type="application/pdf", filename=f"underwriting_report_{report_id[:8]}.pdf")
