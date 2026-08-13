from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _load_model_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"RAG model configuration is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("RAG model configuration must use schema_version 1")
    for section in ("extraction", "embedding", "generation"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"RAG model configuration is missing '{section}'")
    return payload


def _value(section: Mapping[str, Any], key: str, default: Any) -> Any:
    value = section.get(key)
    return default if value in (None, "") else value


def _env_or_config(name: str, section: Mapping[str, Any], key: str, default: str) -> str:
    value = os.getenv(name)
    return (
        value.strip()
        if value not in (None, "")
        else str(_value(section, key, default))
    )


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
    generation_model_id: str = "openai.gpt-oss-20b-1:0"
    generation_temperature: float = 0.1
    generation_max_output_tokens: int = 1800
    chat_retrieval_top_k: int = 8
    chat_max_context_characters: int = 60_000
    extraction_engine: str = "docling"
    layout_model_id: str = "docling-default-layout"
    ocr_model_id: str = "rapidocr-pp-ocrv6"
    table_structure_model_id: str = "tableformer-accurate"
    model_config_path: Path | None = None
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
        config_value = os.getenv("RAG_MODEL_CONFIG_PATH", "").strip()
        model_config_path = (
            Path(config_value).expanduser().resolve()
            if config_value
            else (rag_root / "config" / "models.json").resolve()
        )
        model_config = _load_model_config(model_config_path)
        extraction = model_config["extraction"]
        embedding = model_config["embedding"]
        generation = model_config["generation"]
        credentials_value = os.getenv("RAG_AWS_CREDENTIALS_FILE", "").strip()
        settings = cls(
            data_root=Path(os.getenv("RAG_DATA_ROOT", rag_root / "data")).resolve(),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "sgia_rag_chunks_titan_v2_512"
            ),
            aws_region=os.getenv("AWS_REGION", "eu-central-1"),
            embedding_model_id=_env_or_config(
                "BEDROCK_EMBEDDING_MODEL_ID",
                embedding,
                "model_id",
                "amazon.titan-embed-text-v2:0",
            ),
            embedding_dimensions=_env_int(
                "BEDROCK_EMBEDDING_DIMENSIONS",
                int(_value(embedding, "dimensions", 512)),
            ),
            chunk_max_tokens=_env_int("RAG_CHUNK_MAX_TOKENS", 600),
            chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 60),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
            max_document_pages=_env_int("RAG_MAX_DOCUMENT_PAGES", 500),
            max_extracted_characters=_env_int(
                "RAG_MAX_EXTRACTED_CHARACTERS", 600_000
            ),
            docling_do_ocr=_env_bool(
                "DOCLING_DO_OCR", bool(_value(extraction, "ocr_enabled", True))
            ),
            generation_model_id=_env_or_config(
                "BEDROCK_GENERATION_MODEL_ID",
                generation,
                "model_id",
                "openai.gpt-oss-20b-1:0",
            ),
            generation_temperature=_env_float(
                "RAG_GENERATION_TEMPERATURE",
                float(_value(generation, "temperature", 0.1)),
            ),
            generation_max_output_tokens=_env_int(
                "RAG_GENERATION_MAX_OUTPUT_TOKENS",
                int(_value(generation, "max_output_tokens", 1800)),
            ),
            chat_retrieval_top_k=_env_int("RAG_CHAT_TOP_K", 8),
            chat_max_context_characters=_env_int(
                "RAG_CHAT_MAX_CONTEXT_CHARACTERS", 60_000
            ),
            extraction_engine=str(_value(extraction, "engine", "docling")),
            layout_model_id=str(
                _value(extraction, "layout_model_id", "docling-default-layout")
            ),
            ocr_model_id=str(
                _value(extraction, "ocr_model_id", "rapidocr-pp-ocrv6")
            ),
            table_structure_model_id=str(
                _value(
                    extraction,
                    "table_structure_model_id",
                    "tableformer-accurate",
                )
            ),
            model_config_path=model_config_path,
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
        if not self.generation_model_id:
            raise ValueError("BEDROCK_GENERATION_MODEL_ID must not be blank")
        if not 0 <= self.generation_temperature <= 1:
            raise ValueError("RAG_GENERATION_TEMPERATURE must be between 0 and 1")
        if self.generation_max_output_tokens < 128:
            raise ValueError("RAG_GENERATION_MAX_OUTPUT_TOKENS must be at least 128")
        if not 1 <= self.chat_retrieval_top_k <= 20:
            raise ValueError("RAG_CHAT_TOP_K must be between 1 and 20")
        if self.chat_max_context_characters < 2_000:
            raise ValueError("RAG_CHAT_MAX_CONTEXT_CHARACTERS must be at least 2000")

    def ensure_directories(self) -> None:
        for path in (
            self.uploads_root,
            self.artifacts_root,
            self.jobs_root,
            self.documents_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
