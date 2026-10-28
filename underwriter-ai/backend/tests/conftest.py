"""Shared test fixtures.

Tests run fully offline: SQLite in a temp folder, Qdrant in memory, a deterministic
hashing embedder instead of BGE-M3, no reranker, and no LLM key (the LLM is mocked where
a test needs it).
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session", autouse=True)
def _env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uw")
    os.environ.update({
        "DATA_DIR": str(tmp / "data"), "DATABASE_URL": f"sqlite:///{(tmp / 'test.db').as_posix()}",
        "QDRANT_URL": ":memory:", "RERANKER_ENABLED": "false", "OPENROUTER_API_KEY": "", "OCR_ENABLED": "false",
    })
    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.services.embedding_service import set_embedder

    set_embedder(HashEmbedder())
    yield


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, normalised, good enough for retrieval tests."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _right(page, x_right: float, y: float, text: str, size: float = 9) -> None:
    w = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
    page.insert_text((x_right - w, y), text, fontname="helv", fontsize=size)


def make_sample_drhp(path: Path) -> Path:
    """A 9-page synthetic DRHP: cover, contents, definitions, risks, business, three statements, management."""
    doc = pymupdf.open()

    def page_with(lines: list[str], printed: int | None = None, start_y: float = 72):
        p = doc.new_page(width=595, height=842)
        y = start_y
        for ln in lines:
            p.insert_text((50, y), ln, fontname="helv", fontsize=10)
            y += 14
        if printed is not None:
            p.insert_text((290, 810), str(printed), fontname="helv", fontsize=9)
        return p

    page_with(["DRAFT RED HERRING PROSPECTUS", "Dated: March 1, 2026", "", "ACME WIDGETS LIMITED",
               "Our Company was incorporated as Acme Widgets Private Limited in 2001.", "BOOK RUNNING LEAD MANAGER: XYZ CAPITAL MARKETS LIMITED"])
    page_with(["TABLE OF CONTENTS",
               "SECTION I: GENERAL ..................................... 1",
               "DEFINITIONS AND ABBREVIATIONS ......................... 1",
               "SECTION II: RISK FACTORS .............................. 2",
               "SECTION IV: ABOUT OUR COMPANY ......................... 3",
               "OUR BUSINESS .......................................... 3",
               "SECTION V: FINANCIAL INFORMATION ...................... 4",
               "RESTATED FINANCIAL INFORMATION ........................ 4",
               "SECTION VI: OUR MANAGEMENT ............................ 7",
               "OUR MANAGEMENT ........................................ 7"])
    page_with(["SECTION I: GENERAL", "DEFINITIONS AND ABBREVIATIONS",
               "In this document, Company means Acme Widgets Limited and OFS means offer for sale."], printed=1)
    page_with(["SECTION II: RISK FACTORS",
               "1. We depend on a small number of customers. Our top five customers contributed 62% of revenue in Fiscal 2025.",
               "2. Our manufacturing facility is located in Pune; any disruption could adversely affect our operations.",
               "3. We have outstanding borrowings that contain restrictive covenants."], printed=2)
    page_with(["SECTION IV: ABOUT OUR COMPANY", "OUR BUSINESS",
               "We manufacture precision widgets for the automotive and electronics industries.",
               "Exports to Europe accounted for 18% of revenue from operations in Fiscal 2025."], printed=3)

    def statement(title: str, rows: list[tuple[str, str, str]], printed: int):
        p = page_with(["Acme Widgets Limited", title, "(All amounts in Rs. million)"], printed=printed, start_y=60)
        _right(p, 430, 120, "March 31, 2025")
        _right(p, 530, 120, "March 31, 2024")
        y = 140
        for label, a, b in rows:
            p.insert_text((50, y), label, fontname="helv", fontsize=9)
            if a:
                _right(p, 430, y, a)
                _right(p, 530, y, b)
            y += 14

    statement("Restated Statement of Profit and Loss", [
        ("Revenue from operations", "1,000.00", "800.00"), ("Other income", "20.00", "10.00"),
        ("Total income", "1,020.00", "810.00"), ("Expenses", "", ""), ("Cost of materials consumed", "500.00", "420.00"),
        ("Employee benefits expense", "150.00", "130.00"), ("Finance costs", "30.00", "35.00"),
        ("Depreciation and amortisation expense", "40.00", "38.00"), ("Other expenses", "100.00", "90.00"),
        ("Total expenses", "820.00", "713.00"), ("Profit before tax", "200.00", "97.00"),
        ("Total tax expense", "50.00", "24.00"), ("Profit for the year", "150.00", "73.00"),
        ("Basic", "15.00", "7.30"), ("Diluted", "15.00", "7.30")], printed=4)
    statement("Restated Statement of Assets and Liabilities", [
        ("Inventories", "120.00", "100.00"), ("Trade receivables", "180.00", "150.00"),
        ("Cash and cash equivalents", "60.00", "40.00"), ("Total current assets", "400.00", "320.00"),
        ("Total assets", "1,200.00", "1,050.00"), ("Equity share capital", "100.00", "100.00"),
        ("Other equity", "500.00", "350.00"), ("Total equity", "600.00", "450.00"),
        ("Borrowings", "250.00", "300.00"), ("Total current liabilities", "200.00", "180.00"),
        ("Total liabilities", "600.00", "600.00"), ("Total equity and liabilities", "1,200.00", "1,050.00")], printed=5)
    statement("Restated Statement of Cash Flows", [
        ("Net cash generated from operating activities", "180.00", "120.00"),
        ("Purchase of property, plant and equipment", "(70.00)", "(60.00)"),
        ("Net cash used in investing activities", "(70.00)", "(60.00)"),
        ("Net cash used in financing activities", "(90.00)", "(50.00)"),
        ("Cash and cash equivalents at the end of the year", "60.00", "40.00")], printed=6)
    page_with(["SECTION VI: OUR MANAGEMENT", "OUR MANAGEMENT",
               "Mr. A. Sharma is our Managing Director and has 25 years of experience in manufacturing."], printed=7)
    doc.save(path)
    return path


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return make_sample_drhp(tmp_path_factory.mktemp("pdf") / "acme_drhp.pdf")
