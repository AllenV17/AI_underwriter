"""Request and response models for the REST API."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.valuation.comparables import PeerCompany
from app.valuation.dcf import DCFAssumptions


class DocumentOut(BaseModel):
    id: str
    filename: str
    company_name: str | None
    doc_type: str | None
    n_pages: int | None
    status: str
    stage: str | None
    progress: float
    error: str | None
    indexed: bool
    financials_extracted: bool
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class SectionOut(BaseModel):
    title: str
    key: str
    section: str | None
    pdf_start: int
    pdf_end: int
    printed_start: int
    title_verified: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    shares_outstanding: float | None = Field(default=None, gt=0)
    price_per_share: float | None = Field(default=None, gt=0)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    steps: list[dict]
    citations: list[dict]
    evidence: list[dict]
    llm_used: bool
    compliance_flagged: bool
    warnings: list[str]
    disclaimer: str | None = None


class FinancialValueOut(BaseModel):
    field: str
    label: str
    period: str
    value_original: float | None
    unit: str | None
    value_inr: float | None
    statement: str
    basis: str
    source_page: int | None
    source_label: str
    derived: bool
    formula: str | None


class FinancialsResponse(BaseModel):
    periods: list[str]
    basis: str
    unit: str | None
    values: list[FinancialValueOut]
    checks: list[dict]
    warnings: list[str]
    statement_pages: dict[str, list[int]]


class RatiosResponse(BaseModel):
    ratios: list[dict]
    meta: dict


class ComparableRequest(BaseModel):
    peers: list[PeerCompany] = Field(min_length=1)
    period: str | None = None
    shares_outstanding: float | None = Field(default=None, gt=0, description="Post-issue shares")
    pre_issue_shares: float | None = Field(default=None, gt=0)
    fresh_issue_inr: float | None = Field(default=None, ge=0, description="Fresh-issue proceeds in INR (0 for a pure OFS)")
    overrides: dict[str, float] | None = Field(default=None, description="Override target revenue/ebitda/pat/net_debt (INR)")


class DCFRequest(BaseModel):
    assumptions: DCFAssumptions


class ReportRequest(BaseModel):
    peers: list[PeerCompany] | None = None
    shares_outstanding: float | None = Field(default=None, gt=0)
    pre_issue_shares: float | None = Field(default=None, gt=0)
    fresh_issue_inr: float | None = Field(default=None, ge=0)
    issue_price: float | None = Field(default=None, gt=0)
    dcf_assumptions: DCFAssumptions | None = None


class ReportResponse(BaseModel):
    report_id: str
    report: dict
    pdf_url: str


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    model: str
    database: str
    vector_store: str
    embedding_model: str
    reranker: str
