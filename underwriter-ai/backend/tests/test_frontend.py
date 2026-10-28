"""Smoke-test every Streamlit page headlessly against a running backend.

Skipped unless API_URL points at a live backend with at least one ready document:
    API_URL=http://127.0.0.1:8000 pytest tests/test_frontend.py
"""

import os
from pathlib import Path

import httpx
import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"
API = os.environ.get("API_URL")


def _ready() -> bool:
    if not API:
        return False
    try:
        return any(d["status"] == "ready" for d in httpx.get(f"{API}/api/documents", timeout=5).json())
    except Exception:
        return False


@pytest.mark.skipif(not _ready(), reason="needs API_URL with a ready document")
@pytest.mark.parametrize("page", ["Home", "Document Analysis", "Chat", "Financial Dashboard", "Valuation", "Final Report"])
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    import sys
    sys.path.insert(0, str(FRONTEND.parent))
    at = AppTest.from_file(str(FRONTEND), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
