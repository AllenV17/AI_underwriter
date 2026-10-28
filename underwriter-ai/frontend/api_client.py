"""Thin HTTP client for the UnderwriterAI backend. The frontend never sees API keys."""

import os

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    pass


def _req(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 600), **kw)
    except httpx.HTTPError as e:
        raise APIError(f"Backend not reachable at {API_URL}: {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:300]
        raise APIError(f"{r.status_code}: {detail}")
    return r


def get(path: str, **kw):
    return _req("GET", path, **kw).json()


def post(path: str, **kw):
    return _req("POST", path, **kw).json()


def get_bytes(path: str) -> bytes:
    return _req("GET", path).content
