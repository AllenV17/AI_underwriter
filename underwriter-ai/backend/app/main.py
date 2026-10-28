"""FastAPI application entry point: `uvicorn app.main:app` from the backend/ folder."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.logging import setup_logging
from app.models.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    init_db()
    yield


app = FastAPI(
    title="UnderwriterAI",
    description="AI-assisted DRHP analysis and IPO valuation. Informational and educational use only; not investment advice.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
