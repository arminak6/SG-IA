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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    top_k: int | None = Field(default=None, ge=8, le=10)
    document_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RagCitation(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    source_path: str
    title: str
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    score: float
    excerpt: str


class ChatTimings(BaseModel):
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    verification_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(ge=0)


class ConfidenceDebug(BaseModel):
    enabled: bool = False
    verification_available: bool = False
    model_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    components: dict[str, float | bool] = Field(default_factory=dict)


class ChatDebug(BaseModel):
    requested_top_k: int
    retrieval_strategy: str = "semantic"
    candidate_pool_size: int = 0
    initial_candidate_count: int = 0
    neighbor_candidate_count: int = 0
    final_context_count: int = 0
    retrieval_attempts: int = Field(default=1, ge=1, le=2)
    coverage_facets: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)
    evidence_coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    evidence_coverage_sufficient: bool | None = None
    retrieved_chunks: list[SearchHit] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    generation_attempts: int = Field(default=0, ge=0, le=2)
    generation_stop_reason: str | None = None
    confidence: ConfidenceDebug = Field(default_factory=ConfidenceDebug)
    session_id: str | None = None


class ChatResponse(BaseModel):
    approach: str = "rag"
    status: str
    answer: str
    citations: list[RagCitation] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    timings: ChatTimings
    model_id: str | None = None
    embedding_model_id: str
    confidence_score: float | None = Field(default=None, ge=0, le=10)
    debug: ChatDebug


class ModelConfigurationResponse(BaseModel):
    schema_version: int = 1
    pipeline_version: str = "1.2"
    extraction: dict[str, Any]
    embedding: dict[str, Any]
    chunking: dict[str, Any]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    verification: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    pipeline_version: str = "1.2"
    qdrant: str
    collection: str
    embedding_model_id: str
    generation_model_id: str
    confidence_model_id: str | None = None
