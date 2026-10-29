"""File locations for a document's artefacts, and JSON helpers."""

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.schemas.document_ir import DocumentIR


def upload_path(document_id: str) -> Path:
    return get_settings().uploads_dir / f"{document_id}.pdf"


def processed_dir(document_id: str) -> Path:
    d = get_settings().processed_dir / document_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def ir_path(document_id: str) -> Path:
    return processed_dir(document_id) / "document.json"


def chunks_path(document_id: str) -> Path:
    return processed_dir(document_id) / "chunks.jsonl"


def financials_path(document_id: str) -> Path:
    return processed_dir(document_id) / "financials.json"


def peers_path(document_id: str) -> Path:
    return processed_dir(document_id) / "peers.json"


def save_ir(ir: DocumentIR) -> None:
    ir_path(ir.document_id).write_text(ir.model_dump_json(), encoding="utf-8")


def load_ir(document_id: str) -> DocumentIR | None:
    p = ir_path(document_id)
    return DocumentIR.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def load_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
