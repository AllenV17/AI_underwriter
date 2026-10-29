# UnderwriterAI

AI-assisted analysis of Indian IPO offer documents (DRHP / RHP). Upload a DRHP PDF and the
system maps its sections, extracts and validates the restated financial statements, indexes
every page for cited question answering, computes ratios and valuations deterministically,
and generates an evidence-grounded underwriting report.

> **Not investment advice.** UnderwriterAI is an analysis and educational tool. It is not a
> SEBI-registered investment adviser, research analyst or merchant banker, never gives
> buy/sell/subscribe recommendations, and its valuations are model estimates, not issue or
> listing prices. Regulatory compliance must be reviewed before any public or commercial use.

## Architecture

```
Streamlit frontend ──HTTP──> FastAPI backend
                               ├── ingestion/        PDF validation, PyMuPDF text, OCR fallback, TOC section map
                               ├── financial_engine/ statement locator + parser, units, extraction, ratios
                               ├── rag/              chunker, BM25, Qdrant store, RRF fusion, reranker, citations
                               ├── valuation/        comparables, DCF (+ sensitivity), listing predictor (interface)
                               ├── agent/            tools + orchestrator (intent -> tool plan -> cited answer)
                               ├── services/         llm (OpenRouter), document pipeline, retrieval, report (PDF)
                               └── core/             settings, logging (secret scrubbing), compliance guardrails
PostgreSQL (documents, financial values, chats, reports) · Qdrant (chunk vectors) · data/ (IR, chunks, PDFs)
```

**Retrieval pipeline:** query → BM25 (top 20) + BGE-M3 → Qdrant cosine (top 20) → Reciprocal Rank
Fusion → top 15 → cross-encoder reranker → top 3-5 chunks → LLM. The LLM sees evidence labelled
`[S1]…[S5]`; citations are validated after generation and rewritten to
`(Source: DRHP, <section>, page <n>)` from chunk metadata, so page numbers can never be invented.

**Financial extraction:** the primary statements are found by scoring every page on how many of
the statement's line items it contains (DRHPs repeat statement titles on dozens of note pages),
then tables are rebuilt from word positions: value columns are found by clustering the right
edges of numbers, period headers are read above them, a "Note" column is dropped. Values keep
the printed figure, unit, rupee value, page and row label. Derived figures (EBITDA, net debt,
FCF) carry their formula. Accounting identities are checked automatically.

## Changes from the original specification

| Change | Why |
|---|---|
| Section detection uses the DRHP's printed table of contents (PyMuPDF), not Docling | SEBI's ICDR order makes the contents page reliable; this mapped 358/358 chapters on the 10 pilot DRHPs, in seconds |
| Docling is used for tables on selected chapters (`TABLE_ENGINE=docling`) and as an OCR fallback, not on all pages | A layout model over 500 pages takes many minutes on CPU |
| Financial statements are parsed by a dedicated word-position parser | DRHP statements are borderless; ruled-table extractors merge or split their columns |
| Default reranker `BAAI/bge-reranker-base` (configurable to `bge-reranker-v2-m3`) | Several times faster on CPU |
| Embedded Qdrant + SQLite fallback when no servers are configured | Runs and tests without Docker |
| Agent orchestration is an explicit state graph instead of LangGraph | Same control flow, deterministic and unit-testable intent routing, no model-specific function-calling dependency |
| Lender detection (finance costs > 35% of revenue) | EBITDA, interest cover and current ratios are not meaningful for NBFCs/HFCs and are marked n/a with the reason |
| Compliance module (disclaimer, advice-request detection, output screening) | Implements the SEBI-related guardrails in one place |

## Quick start

### Docker Compose
```bash
cp .env.example .env          # add OPENROUTER_API_KEY and OPENROUTER_MODEL
docker compose up --build
```
Frontend: http://localhost:8501 · API docs: http://localhost:8000/docs. The first document
processed downloads BGE-M3 (~2.3 GB) and the reranker into the `models` volume.

### Local (no Docker)
```bash
python -m venv .venv && .venv/Scripts/activate          # Windows; use bin/activate on Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
cp .env.example .env    # set the OpenRouter key; delete QDRANT_URL/DATABASE_URL lines for embedded mode
cd backend && uvicorn app.main:app --port 8000
# second terminal
streamlit run frontend/streamlit_app.py
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | – | LLM access (never sent to the frontend or logs) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Any OpenRouter model id |
| `DATABASE_URL` | SQLite under `data/` | PostgreSQL in Docker |
| `QDRANT_URL` | embedded folder under `data/` | `http://host:6333` for a server, `:memory:` for tests |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Used for chunks and queries |
| `RERANKER_MODEL` / `RERANKER_ENABLED` | `BAAI/bge-reranker-base` / true | Cross-encoder reranking |
| `TABLE_ENGINE` | `pdfplumber` | `docling` for layout-model tables |
| `OCR_ENABLED` | true | Tesseract (if installed) or Docling OCR for image-only pages |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service, DB, LLM status |
| POST | `/api/documents/upload` | Upload a PDF (processing starts automatically) |
| POST | `/api/documents/{id}/process` | (Re)process a document |
| GET | `/api/documents`, `/api/documents/{id}` | List / status and progress |
| GET | `/api/documents/{id}/sections` | Chapter map with page ranges |
| POST | `/api/documents/{id}/chat` | Agent answer with citations and retrieved evidence |
| GET | `/api/documents/{id}/financials` | Extracted values, sources, validation checks |
| GET | `/api/documents/{id}/peers` | Peer table printed in the DRHP's Basis for Offer Price (suggestions to verify) |
| GET | `/api/documents/{id}/ratios` | Ratios (optional `price_per_share`, `shares_outstanding`) |
| POST | `/api/documents/{id}/valuation/comparable` | Peer-multiple valuation (user-supplied peers) |
| GET | `/api/documents/{id}/valuation/dcf/defaults` | DRHP base-year inputs for the DCF |
| POST | `/api/documents/{id}/valuation/dcf` | DCF + sensitivity tables |
| POST | `/api/documents/{id}/report` | Full underwriting report (JSON + PDF) |
| GET | `/api/documents/{id}/report/{report_id}/pdf` | Download the PDF |
| GET | `/api/compliance/disclaimer` | Disclaimer text |

## Tests
```bash
cd backend && ../.venv/Scripts/python -m pytest
```
Unit and API tests run offline (fake embedder, in-memory Qdrant, mocked LLM) on a synthetic DRHP.
`tests/test_pilot_drhps.py` checks extraction against figures printed in the real pilot DRHPs
(`../data/raw/drhp/`) and is skipped when they are absent.

## Phase status

| Phase | Status |
|---|---|
| 1 Foundation (API, frontend, OpenRouter, upload, parsing) | Done |
| 2 RAG (chunking, BGE-M3, Qdrant, BM25, RRF, reranking, cited QA) | Done |
| 3 Financial analysis (extraction, normalisation, ratios, dashboard) | Done; see limitations |
| 4 Valuation (comparables, DCF, sensitivity, range) | Done; peers are user-supplied. Peer multiples give pre-money equity, so enter the pre-issue share count or the fresh-issue proceeds (0 for a pure OFS); with only a post-issue count the per-share value is understated and a warning says so |
| 5 Agent (tools, orchestration, report generation, PDF) | Done |
| 6 Historical IPO prediction | Training, walk-forward evaluation and predictor are implemented. `data/ipo_history.csv` has 129 priced IPOs but pre-listing features exist only for the 8 pilots, so only a subscription-only (post-bidding) model can be trained, and it does not yet beat naive baselines on the newest IPOs. The predictor returns no number unless a model has been trained and validated (statuses `not_trained`, `not_validated`, `insufficient_inputs`) |

## Known limitations
- Extraction is validated on 10 pilot DRHPs (see `Evaluation_Results.md`); unusual layouts can still leave fields missing. Missing values are reported as missing, never zero.
- The DRHP's own peer P/E table is read automatically (`GET /api/documents/{id}/peers`) as a suggestion to verify; wrapped company names can be mis-read and some tables are blank. Peer market data and share counts are otherwise user inputs; nothing is fetched from exchanges.
- Risk severity is an LLM assessment of the disclosed text; every risk must cite DRHP evidence or it is dropped.
- BGE-M3 on CPU is slow: about 25 minutes to embed a 361-page DRHP (~1,200 chunks) on a laptop CPU.
  A GPU brings this to about a minute. For quick local trials set `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
  (much faster, English-only, lower recall); re-process documents after changing the model.
  Chat and the report become available once processing (including embedding) has finished.
