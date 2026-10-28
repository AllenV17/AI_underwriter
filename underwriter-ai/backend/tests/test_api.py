"""End-to-end API tests on the synthetic DRHP (offline: fake embedder, in-memory Qdrant, no LLM key)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def doc_id(client, sample_pdf):
    with open(sample_pdf, "rb") as f:
        r = client.post("/api/documents/upload", files={"file": ("acme_drhp.pdf", f, "application/pdf")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    # TestClient runs background tasks before returning, so processing has finished.
    d = client.get(f"/api/documents/{did}").json()
    assert d["status"] == "ready", d
    return did


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["database"] == "ok" and r["llm_configured"] is False


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/documents/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_document_processed(client, doc_id):
    d = client.get(f"/api/documents/{doc_id}").json()
    assert d["company_name"] == "Acme Widgets Limited" and d["doc_type"] == "DRHP"
    assert d["indexed"] and d["financials_extracted"]
    secs = client.get(f"/api/documents/{doc_id}/sections").json()
    assert {"risk_factors", "our_business", "restated_financials"} <= {s["key"] for s in secs}


def test_financials_and_ratios(client, doc_id):
    f = client.get(f"/api/documents/{doc_id}/financials").json()
    assert f["periods"] == ["FY2025", "FY2024"]
    rev = next(v for v in f["values"] if v["field"] == "revenue_from_operations" and v["period"] == "FY2025")
    assert rev["value_inr"] == pytest.approx(1e9) and rev["source_page"] == 6
    r = client.get(f"/api/documents/{doc_id}/ratios", params={"price_per_share": 300, "shares_outstanding": 1e7}).json()
    pe = next(x for x in r["ratios"] if x["name"] == "P/E" and x["period"] == "FY2025")
    assert pe["value"] == pytest.approx(20.0)


def test_chat_without_llm_returns_evidence(client, doc_id):
    r = client.post(f"/api/documents/{doc_id}/chat", json={"question": "What are the risks of customer concentration?"}).json()
    assert r["intent"] == "risk"
    assert not r["llm_used"] and "not configured" in r["answer"]


def test_chat_with_mocked_llm_cites_real_pages(client, doc_id, monkeypatch):
    from app.services import llm_service

    def fake_chat(self, system, user, **kw):
        return llm_service.LLMResponse(text="Top five customers contributed 62% of revenue [S1]. Also [S99].", model="m")

    monkeypatch.setattr(llm_service.LLMService, "chat", fake_chat)
    r = client.post(f"/api/documents/{doc_id}/chat", json={"question": "Who are the key customers and how concentrated is revenue?"}).json()
    assert r["llm_used"]
    assert "(Source: DRHP," in r["answer"] and "S99" not in r["answer"]
    assert any("S99" in w for w in r["warnings"])


def test_advice_question_is_refused_personalised(client, doc_id, monkeypatch):
    from app.services import llm_service

    monkeypatch.setattr(llm_service.LLMService, "chat",
                        lambda self, s, u, **k: llm_service.LLMResponse(text="You should subscribe to this IPO [S1].", model="m"))
    r = client.post(f"/api/documents/{doc_id}/chat", json={"question": "Should I subscribe to this IPO?"}).json()
    assert r["answer"].startswith("I can't give a personalised recommendation")
    assert r["compliance_flagged"]


def test_comparable_and_dcf(client, doc_id):
    peers = [{"name": "Peer A", "pe": 20, "ps": 3, "source": "test"}, {"name": "Peer B", "pe": 24, "ev_ebitda": 14, "source": "test"},
             {"name": "Peer C", "pe": 18, "ev_ebitda": 12, "source": "test"}]
    r = client.post(f"/api/documents/{doc_id}/valuation/comparable", json={"peers": peers, "shares_outstanding": 1e7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"]["period"] == "FY2025" and body["price_range"] and "not the official IPO issue price" in body["notice"]
    defaults = client.get(f"/api/documents/{doc_id}/valuation/dcf/defaults").json()
    assert defaults["base_revenue"] == pytest.approx(1e9)
    a = {"base_revenue": defaults["base_revenue"], "revenue_growth": [0.12, 0.1, 0.08], "ebitda_margin": 0.25, "wacc": 0.12,
         "terminal_growth": 0.05, "net_debt": defaults["net_debt"], "shares_outstanding": 1e7}
    d = client.post(f"/api/documents/{doc_id}/valuation/dcf", json={"assumptions": a}).json()
    assert d["price_per_share"] > 0 and d["sensitivity_wacc_g"]["values"]


def test_report_pdf(client, doc_id):
    r = client.post(f"/api/documents/{doc_id}/report", json={"shares_outstanding": 1e7})
    assert r.status_code == 200, r.text
    body = r.json()
    titles = [s["title"] for s in body["report"]["sections"]]
    for t in ("Executive Summary", "Financial Performance", "Risk Factor Analysis", "Regulatory and Investment Disclaimer",
              "Sources and Citations", "Potential Listing-Price Scenario"):
        assert t in titles
    pdf = client.get(body["pdf_url"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
