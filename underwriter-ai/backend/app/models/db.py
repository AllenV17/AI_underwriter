"""Relational storage (PostgreSQL in Docker, SQLite for local runs and tests).

Holds structured application data: documents and their processing status, extracted
financial values, chat history and generated reports. Large artefacts (page text, chunks,
tables) live as files under data/processed/<document_id>/.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.core.config import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    company_name: Mapped[str | None] = mapped_column(String(255))
    doc_type: Mapped[str | None] = mapped_column(String(32))
    n_pages: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded/processing/ready/failed
    stage: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    financials_extracted: Mapped[bool] = mapped_column(Boolean, default=False)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    financial_values: Mapped[list["FinancialValueRow"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class FinancialValueRow(Base):
    __tablename__ = "financial_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(255))
    period: Mapped[str] = mapped_column(String(32))
    value_original: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(16))
    value_inr: Mapped[float | None] = mapped_column(Float)
    statement: Mapped[str] = mapped_column(String(16))
    basis: Mapped[str] = mapped_column(String(16))
    source_page: Mapped[int | None] = mapped_column(Integer)
    source_label: Mapped[str] = mapped_column(Text)
    derived: Mapped[bool] = mapped_column(Boolean, default=False)
    formula: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="financial_values")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[dict] = mapped_column(JSON)
    pdf_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@lru_cache
def _engine_for(url: str):
    # "postgresql://" defaults to psycopg2 in SQLAlchemy; this project ships psycopg 3.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kwargs)


def get_engine():
    return _engine_for(get_settings().database_url)


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_factory()() as s:
        yield s
