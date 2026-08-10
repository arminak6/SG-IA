from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ElementType(str, Enum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    PICTURE = "picture"
    FORMULA = "formula"
    CODE = "code"


class BoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class DocumentElement(BaseModel):
    element_id: str
    element_type: ElementType
    text: str
    page_number: int | None = None
    bounding_box: BoundingBox | None = None
    heading_path: list[str] = Field(default_factory=list)
    artifact_path: str | None = None


class ExtractionResult(BaseModel):
    parser: str
    parser_version: str | None = None
    page_count: int | None = None
    elements: list[DocumentElement]
    warnings: list[str] = Field(default_factory=list)


class RagChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_hash: str
    filename: str
    title: str
    ordinal: int
    text: str
    embedding_text: str
    token_count: int
    element_ids: list[str]
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    content_types: list[ElementType] = Field(default_factory=list)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str, Enum):
    INDEXED = "indexed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    job_id: str
    document_id: str
    filename: str
    source_hash: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    duplicate: bool = False
    chunk_count: int | None = None
    error: str | None = None


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    title: str
    source_hash: str
    media_type: str | None = None
    status: DocumentStatus = DocumentStatus.INDEXED
    created_at: datetime = Field(default_factory=utc_now)
    indexed_at: datetime = Field(default_factory=utc_now)
    parser: str
    parser_version: str | None = None
    page_count: int | None = None
    element_count: int
    chunk_count: int
    embedding_model_id: str
    embedding_dimensions: int
    warnings: list[str] = Field(default_factory=list)


class UploadAccepted(BaseModel):
    job: IngestionJob
    duplicate: bool
    message: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    title: str
    score: float
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float
    embedding_model_id: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
    embedding_model_id: str
