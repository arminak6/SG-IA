"""Pydantic request and response models for the LLM Wiki HTTP API."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    """Base model with a strict, stable public API shape."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str
    bedrock_configured: bool
    model_id: Optional[str] = None
    region: Optional[str] = None


class DocumentResponse(ApiModel):
    relative_path: str
    status: str
    size_bytes: int = Field(ge=0)
    modified_at: datetime


class DocumentsResponse(ApiModel):
    documents: List[DocumentResponse] = Field(default_factory=list)


class UpdateWikiRequest(ApiModel):
    paths: Optional[List[str]] = None

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, paths: Optional[List[str]]) -> Optional[List[str]]:
        if paths is None:
            return None

        normalized_paths: List[str] = []
        seen = set()
        for raw_path in paths:
            path = raw_path.strip().replace("\\", "/")
            if not path:
                raise ValueError("document paths cannot be empty")

            posix_path = PurePosixPath(path)
            windows_path = PureWindowsPath(path)
            if posix_path.is_absolute() or windows_path.is_absolute() or ".." in posix_path.parts:
                raise ValueError("document paths must be relative and cannot contain '..'")

            normalized = posix_path.as_posix()
            if normalized not in seen:
                normalized_paths.append(normalized)
                seen.add(normalized)

        return normalized_paths


class FailedDocumentResponse(ApiModel):
    path: str
    error: str


class UpdateWikiResponse(ApiModel):
    processed: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)
    failed: List[FailedDocumentResponse] = Field(default_factory=list)


class ChatRequest(ApiModel):
    question: str = Field(min_length=1, max_length=10_000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be blank")
        return question


class CitationResponse(ApiModel):
    wiki_path: str
    source_paths: List[str] = Field(default_factory=list)


class ChatDebugResponse(ApiModel):
    pages_read: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    search_modes: List[str] = Field(default_factory=list)
    retrieval_diagnostics: List[dict[str, object]] = Field(default_factory=list)


class ChatResponse(ApiModel):
    approach: str = "wiki"
    status: str
    answer: str
    citations: List[CitationResponse] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    model_id: Optional[str] = None
    debug: ChatDebugResponse = Field(default_factory=ChatDebugResponse)


class WikiPageResponse(ApiModel):
    relative_path: str
    title: Optional[str] = None
    summary: Optional[str] = None
    source_paths: List[str] = Field(default_factory=list)
    size_bytes: Optional[int] = Field(default=None, ge=0)
    modified_at: Optional[datetime] = None


class WikiPagesResponse(ApiModel):
    pages: List[WikiPageResponse] = Field(default_factory=list)


class WikiLintIssue(ApiModel):
    severity: str
    code: str
    path: str
    message: str


class WikiGraphSummaryResponse(ApiModel):
    pages: int = Field(ge=0)
    links: int = Field(ge=0)
    without_incoming: int = Field(ge=0)
    without_outgoing: int = Field(ge=0)
    isolated: int = Field(ge=0)


class WikiLintResponse(ApiModel):
    valid: bool
    pages_checked: int = Field(ge=0)
    issues: List[WikiLintIssue] = Field(default_factory=list)
    graph: WikiGraphSummaryResponse


class LinkRepairRequest(ApiModel):
    max_links: int = Field(default=12, ge=1, le=50)


class LinkProposalResponse(ApiModel):
    source_path: str
    target_path: str
    reason: str


class LinkRepairResponse(ApiModel):
    links_added: List[LinkProposalResponse] = Field(default_factory=list)
    pages_updated: List[str] = Field(default_factory=list)
    graph_before: WikiGraphSummaryResponse
    graph_after: WikiGraphSummaryResponse
    usage: dict[str, int] = Field(default_factory=dict)
    warning: Optional[str] = None


class ErrorResponse(ApiModel):
    detail: str


def model_to_dict(value: Any) -> dict[str, Any]:
    """Convert common service return types into a plain mapping."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    try:
        return vars(value)
    except TypeError as exc:  # pragma: no cover - defensive boundary
        raise TypeError(f"Expected a mapping-like service result, got {type(value).__name__}") from exc
