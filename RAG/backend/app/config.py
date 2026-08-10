from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    qdrant_url: str
    qdrant_collection: str
    aws_region: str
    embedding_model_id: str
    embedding_dimensions: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    max_upload_bytes: int
    max_document_pages: int
    max_extracted_characters: int
    docling_do_ocr: bool
    credentials_file: Path | None = None

    @property
    def uploads_root(self) -> Path:
        return self.data_root / "uploads"

    @property
    def artifacts_root(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def documents_root(self) -> Path:
        return self.data_root / "documents"

    @classmethod
    def from_env(cls) -> Settings:
        rag_root = Path(__file__).resolve().parents[2]
        credentials_value = os.getenv("RAG_AWS_CREDENTIALS_FILE", "").strip()
        settings = cls(
            data_root=Path(os.getenv("RAG_DATA_ROOT", rag_root / "data")).resolve(),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "sgia_rag_chunks_titan_v2_512"
            ),
            aws_region=os.getenv("AWS_REGION", "eu-central-1"),
            embedding_model_id=os.getenv(
                "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
            ),
            embedding_dimensions=_env_int("BEDROCK_EMBEDDING_DIMENSIONS", 512),
            chunk_max_tokens=_env_int("RAG_CHUNK_MAX_TOKENS", 600),
            chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 60),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
            max_document_pages=_env_int("RAG_MAX_DOCUMENT_PAGES", 500),
            max_extracted_characters=_env_int(
                "RAG_MAX_EXTRACTED_CHARACTERS", 600_000
            ),
            docling_do_ocr=_env_bool("DOCLING_DO_OCR", True),
            credentials_file=(
                Path(credentials_value).expanduser().resolve()
                if credentials_value
                else None
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.embedding_dimensions not in {256, 512, 1024}:
            raise ValueError("BEDROCK_EMBEDDING_DIMENSIONS must be 256, 512, or 1024")
        if self.chunk_max_tokens < 50:
            raise ValueError("RAG_CHUNK_MAX_TOKENS must be at least 50")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_max_tokens:
            raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be >= 0 and below max")

    def ensure_directories(self) -> None:
        for path in (
            self.uploads_root,
            self.artifacts_root,
            self.jobs_root,
            self.documents_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

