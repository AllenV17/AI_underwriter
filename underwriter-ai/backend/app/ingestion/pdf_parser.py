"""PDF validation and page-level text extraction.

PyMuPDF extracts page text (fast and robust on 300-700 page filings). Pages with no text
layer, such as scanned auditor reports, are sent to OCR: PyMuPDF's Tesseract
integration when Tesseract is installed, otherwise Docling's OCR pipeline for those pages.
If neither is available the page is kept empty and a warning is recorded, never guessed.
"""

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 40  # below this a page is treated as image-only


class InvalidPDFError(ValueError):
    """The upload is not a readable PDF."""


@dataclass
class RawPage:
    number: int  # 1-based
    text: str
    ocr: bool = False
    headings: list[str] = field(default_factory=list)


@dataclass
class RawDocument:
    pages: list[RawPage]
    warnings: list[str] = field(default_factory=list)


_MOJIBAKE = {"â€¢": "•", "â€“": "–", "â€”": "—", "â€™": "'", "â€˜": "'", "â€œ": '"', "â€": '"', "Â ": " "}  # UTF-8 read as cp1252


def clean_text(text: str) -> str:
    """Drop invalid Unicode (lone surrogates from broken font maps) and fix common UTF-8 mojibake."""
    text = text.encode("utf-8", "ignore").decode("utf-8").replace("�", "")
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    return text


def validate_pdf(path: Path, max_mb: int = 150) -> int:
    """Check the file is a real, non-encrypted PDF within the size limit. Returns page count."""
    size_mb = path.stat().st_size / 1_048_576
    if size_mb > max_mb:
        raise InvalidPDFError(f"File is {size_mb:.0f} MB; the limit is {max_mb} MB.")
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise InvalidPDFError("File is not a PDF (missing %PDF header).")
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # pymupdf raises various internal errors on corrupt files
        raise InvalidPDFError(f"PDF could not be opened: {e}") from e
    with doc:
        if doc.needs_pass:
            raise InvalidPDFError("PDF is password-protected.")
        if len(doc) == 0:
            raise InvalidPDFError("PDF has no pages.")
        return len(doc)


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _headings(page: "pymupdf.Page") -> list[str]:
    """Short bold lines on the page, in reading order: candidate sub-section headings."""
    out = []
    try:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                if bold and 4 <= len(text) <= 90 and not re.fullmatch(r"[\d\s.,()%₹-]+", text):
                    out.append(re.sub(r"\s+", " ", text))
    except Exception:  # heading detection is best effort
        return []
    return out


def extract_pages(path: Path, ocr_enabled: bool = True) -> RawDocument:
    """Extract text for every page, with OCR for image-only pages when available."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    warnings: list[str] = []
    pages: list[RawPage] = []
    image_only: list[int] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text", sort=True))
            rp = RawPage(number=i + 1, text=text, headings=_headings(page))
            if len(text.strip()) < MIN_TEXT_CHARS and page.get_images():
                image_only.append(i)
            pages.append(rp)

        if image_only and ocr_enabled:
            if _tesseract_available():
                for i in image_only:
                    try:
                        tp = doc[i].get_textpage_ocr(language="eng", dpi=200, full=True)
                        pages[i].text = clean_text(doc[i].get_text("text", textpage=tp, sort=True))
                        pages[i].ocr = True
                    except Exception as e:
                        warnings.append(f"OCR failed on page {i + 1}: {e}")
            else:
                done = _docling_ocr(path, [i + 1 for i in image_only], pages)
                if not done:
                    warnings.append(
                        f"{len(image_only)} image-only page(s) without OCR (Tesseract/Docling OCR unavailable): "
                        f"{[i + 1 for i in image_only][:20]}")
        elif image_only:
            warnings.append(f"{len(image_only)} image-only page(s) skipped because OCR is disabled.")
    return RawDocument(pages=pages, warnings=warnings)


def _docling_ocr(path: Path, page_numbers: list[int], pages: list[RawPage]) -> bool:
    """OCR selected pages with Docling. Returns False if Docling is not usable."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        return False
    try:
        opts = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
        opts.ocr_options.force_full_page_ocr = True
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        for n in page_numbers:
            res = conv.convert(str(path), page_range=(n, n))
            text = res.document.export_to_text()
            if text.strip():
                pages[n - 1].text = text
                pages[n - 1].ocr = True
        return True
    except Exception as e:
        log.warning("Docling OCR failed: %s", e)
        return False


_DOC_TYPES = [("DRAFT RED HERRING PROSPECTUS", "DRHP"), ("RED HERRING PROSPECTUS", "RHP"), ("PROSPECTUS", "Prospectus")]


def detect_doc_type(cover_text: str) -> str | None:
    up = cover_text.upper()
    for needle, label in _DOC_TYPES:
        if needle in up:
            return label
    return None


_COMPANY = re.compile(r"\b([A-Z][A-Z0-9&.,'()\- ]{2,80}?\s+(?:PRIVATE\s+)?LIMITED)\b")
_NOT_COMPANY = re.compile(r"(MERCHANT|BANKERS?|REGISTRAR|CAPITAL MARKETS|SECURITIES|BOOK RUNNING|LEAD MANAGER|"
                          r"KFIN|LINK INTIME|STOCK EXCHANGE|DEPOSITORY|AUDITORS?)")


def detect_company_name(cover_text: str) -> str | None:
    """The issuer's name from the cover page: the first upper-case '... LIMITED' that is not an intermediary."""
    for m in _COMPANY.finditer(cover_text):
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.-")
        name = re.sub(r"^(OF|THE|BY|AND)\s+", "", name)
        if _NOT_COMPANY.search(name) or len(name.split()) > 9:
            continue
        # Prefer the name as printed in mixed case elsewhere on the cover ("Unicommerce eSolutions Limited").
        pattern = re.escape(name).replace(r"\ ", r"\s+")
        for m2 in re.finditer(pattern, cover_text, re.I):
            if not m2.group(0).isupper():
                return re.sub(r"\s+", " ", m2.group(0))
        return name.title()
    return None
