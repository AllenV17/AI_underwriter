"""Application settings, loaded from environment variables (and an optional .env file).

Every external dependency (LLM, vector store, database, models) is configured here so
that no module hard-codes a URL, model name or secret.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# underwriter-ai/ (the project root): backend/app/core/config.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenRouter, OpenAI-compatible) ---
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 4
    llm_temperature: float = 0.1

    # --- Storage ---
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'underwriter.db').as_posix()}"
    # "http://localhost:6333" for a Qdrant server; ":memory:" or a folder path for embedded mode.
    qdrant_url: str = str(PROJECT_ROOT / "data" / "qdrant")
    qdrant_collection: str = "drhp_chunks"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- RAG models ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True

    # --- Retrieval parameters ---
    retrieval_dense_k: int = 20
    retrieval_sparse_k: int = 20
    retrieval_fused_k: int = 15
    retrieval_final_k: int = 5
    rrf_k: int = 60

    # --- Document processing ---
    max_upload_mb: int = 150
    ocr_enabled: bool = True
    # "pdfplumber" (fast, default) or "docling" (layout model, slower) for financial tables.
    table_engine: str = Field(default="pdfplumber", pattern="^(pdfplumber|docling)$")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance; tests clear the cache to override values."""
    s = Settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    return s
