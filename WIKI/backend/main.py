"""FastAPI entry point for the LLM Wiki backend.

Run from the repository root with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from threading import Lock
from typing import Any, Iterable, Mapping, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:  # Allows both ``uvicorn backend.main:app`` and direct module execution.
    from .api_models import (
        ChatRequest,
        ChatResetRequest,
        ChatResetResponse,
        ChatResponse,
        CitationResponse,
        DocumentResponse,
        DocumentsResponse,
        FailedDocumentResponse,
        HealthResponse,
        LinkProposalResponse,
        LinkRepairRequest,
        LinkRepairResponse,
        UpdateWikiRequest,
        UpdateWikiResponse,
        UserProfileResponse,
        UserProfileUpdateRequest,
        WikiPageResponse,
        WikiPagesResponse,
        WikiLintIssue,
        WikiLintResponse,
        WikiGraphSummaryResponse,
        model_to_dict,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from api_models import (  # type: ignore[no-redef]
        ChatRequest,
        ChatResetRequest,
        ChatResetResponse,
        ChatResponse,
        CitationResponse,
        DocumentResponse,
        DocumentsResponse,
        FailedDocumentResponse,
        HealthResponse,
        LinkProposalResponse,
        LinkRepairRequest,
        LinkRepairResponse,
        UpdateWikiRequest,
        UpdateWikiResponse,
        UserProfileResponse,
        UserProfileUpdateRequest,
        WikiPageResponse,
        WikiPagesResponse,
        WikiLintIssue,
        WikiLintResponse,
        WikiGraphSummaryResponse,
        model_to_dict,
    )


logger = logging.getLogger(__name__)


app = FastAPI(
    title="LLM Wiki API",
    version="0.1.0",
    description="Ingest raw documents into an LLM-maintained wiki and ask grounded questions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


_update_lock = Lock()


def _create_service() -> Any:
    """Load the core service without making this API module import-order sensitive."""

    package = __package__ or "backend"
    candidates = (
        f"{package}.app.service",
        f"{package}.service",
        f"{package}.wiki_service",
    )
    import_errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only skip a missing candidate. A missing dependency *inside* a real
            # service module is actionable and should not be disguised.
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                import_errors.append(module_name)
                continue
            raise RuntimeError(
                f"The wiki service could not start because dependency '{exc.name}' is missing."
            ) from exc

        for factory_name in ("get_service", "create_service"):
            factory = getattr(module, factory_name, None)
            if callable(factory):
                return factory()

        service_class = getattr(module, "WikiService", None)
        if service_class is not None:
            return service_class()

    checked = ", ".join(import_errors)
    raise RuntimeError(f"Wiki service implementation is unavailable (checked: {checked}).")


@lru_cache(maxsize=1)
def get_service() -> Any:
    """Return the process-wide service; override this dependency in API tests."""

    return _create_service()


def _service_error(operation: str, exc: Exception) -> HTTPException:
    """Map known domain errors to useful responses without leaking credentials."""

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc) or "Resource not found")
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc) or "Invalid request")

    # Exception messages from SDKs can contain request details. Log only the
    # exception type at this boundary; the service may emit its own redacted log.
    logger.error(
        "Wiki service operation '%s' failed (%s)", operation, type(exc).__name__
    )
    return HTTPException(status_code=503, detail=f"Wiki service could not {operation}.")


def _path_from_item(value: Any) -> str:
    if isinstance(value, str):
        return value
    item = model_to_dict(value)
    for key in ("source_path", "path", "relative_path", "source"):
        if key in item:
            return str(item[key])
    return str(value)


def _normalize_citation(value: Any) -> CitationResponse:
    if isinstance(value, str):
        return CitationResponse(wiki_path=value, source_paths=[])
    if isinstance(value, Mapping):
        wiki_path = None
        for key in ("wiki_path", "citation", "source", "relative_path", "path"):
            if value.get(key):
                wiki_path = str(value[key])
                break
        raw_sources = value.get("source_paths", [])
        if wiki_path:
            return CitationResponse(
                wiki_path=wiki_path,
                source_paths=(
                    [str(source) for source in raw_sources]
                    if isinstance(raw_sources, list)
                    else []
                ),
            )
    return CitationResponse(wiki_path=str(value), source_paths=[])


def _normalize_failed(values: Any) -> list[FailedDocumentResponse]:
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = [{"path": path, "error": error} for path, error in values.items()]

    failed = []
    for value in values:
        if isinstance(value, str):
            failed.append(FailedDocumentResponse(path=value, error="Unknown ingestion error"))
            continue
        item = model_to_dict(value)
        failed.append(
            FailedDocumentResponse(
                path=str(
                    item.get(
                        "source_path", item.get("path", item.get("relative_path", "unknown"))
                    )
                ),
                error=str(item.get("error", item.get("message", "Unknown ingestion error"))),
            )
        )
    return failed


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: Any = Depends(get_service)) -> HealthResponse:
    try:
        result = model_to_dict(service.health())
        bedrock = result.get("bedrock", {})
        return HealthResponse(
            status=str(result.get("status", "ok")),
            bedrock_configured=bool(
                result.get("bedrock_configured", bedrock.get("configured", False))
            ),
            model_id=result.get("model_id", bedrock.get("model_id")),
            region=result.get("region", bedrock.get("region_name")),
        )
    except Exception as exc:
        raise _service_error("report health", exc) from exc


@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents(service: Any = Depends(get_service)) -> DocumentsResponse:
    try:
        result = service.list_documents()
        values: Iterable[Any]
        if isinstance(result, Mapping):
            values = result.get("documents", [])
        else:
            values = result
        documents = []
        for value in values:
            item = model_to_dict(value)
            documents.append(
                DocumentResponse(
                    relative_path=str(item["relative_path"]),
                    status=str(item["status"]),
                    size_bytes=int(item["size_bytes"]),
                    modified_at=item["modified_at"],
                )
            )
        return DocumentsResponse(documents=documents)
    except Exception as exc:
        raise _service_error("list documents", exc) from exc


@app.post("/wiki/update", response_model=UpdateWikiResponse, tags=["wiki"])
def update_wiki(
    request: Optional[UpdateWikiRequest] = None,
    service: Any = Depends(get_service),
) -> UpdateWikiResponse:
    paths = request.paths if request is not None else None
    try:
        # A process should run only one wiki-writing batch at a time. The core
        # service still processes each source in the batch sequentially.
        with _update_lock:
            result = service.update_wiki(paths)
        if isinstance(result, UpdateWikiResponse):
            return result
        payload = model_to_dict(result)
        return UpdateWikiResponse(
            processed=[_path_from_item(item) for item in payload.get("processed", [])],
            skipped=[_path_from_item(item) for item in payload.get("skipped", [])],
            failed=_normalize_failed(payload.get("failed", [])),
        )
    except Exception as exc:
        raise _service_error("update the wiki", exc) from exc


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest, service: Any = Depends(get_service)) -> ChatResponse:
    try:
        result = service.ask(
            request.question,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        if isinstance(result, str):
            return ChatResponse(
                approach="wiki",
                status="answered",
                answer=result,
                citations=[],
                latency_ms=0,
            )
        payload = model_to_dict(result)
        return ChatResponse(
            approach=str(payload.get("approach", "wiki")),
            status=str(payload.get("status", "answered")),
            answer=str(payload.get("answer", "")),
            citations=[_normalize_citation(item) for item in payload.get("citations", [])],
            usage={str(key): int(value) for key, value in payload.get("usage", {}).items()},
            latency_ms=float(payload.get("latency_ms", 0)),
            model_id=payload.get("model_id"),
            confidence_score=payload.get("confidence_score"),
            preference_changed=bool(payload.get("preference_changed", False)),
            preference_operation=str(payload.get("preference_operation", "none")),
            debug=payload.get("debug", {}),
            manager_action=payload.get("manager_action"),
            correction=payload.get("correction"),
        )
    except Exception as exc:
        raise _service_error("answer the question", exc) from exc


@app.get(
    "/users/profile",
    response_model=UserProfileResponse,
    tags=["users"],
)
def get_user_profile(
    user_id: str,
    service: Any = Depends(get_service),
) -> UserProfileResponse:
    try:
        return UserProfileResponse(**model_to_dict(service.get_user_profile(user_id)))
    except Exception as exc:
        raise _service_error("load the user profile", exc) from exc


@app.put(
    "/users/profile",
    response_model=UserProfileResponse,
    tags=["users"],
)
def update_user_profile(
    request: UserProfileUpdateRequest,
    service: Any = Depends(get_service),
) -> UserProfileResponse:
    try:
        return UserProfileResponse(
            **model_to_dict(
                service.update_user_profile(request.user_id, request.preferences)
            )
        )
    except Exception as exc:
        raise _service_error("update the user profile", exc) from exc


@app.post(
    "/chat/reset",
    response_model=ChatResetResponse,
    tags=["chat"],
)
def reset_chat(
    request: ChatResetRequest,
    service: Any = Depends(get_service),
) -> ChatResetResponse:
    try:
        return ChatResetResponse(
            **model_to_dict(service.reset_chat(request.user_id, request.session_id))
        )
    except Exception as exc:
        raise _service_error("reset the chat session", exc) from exc


@app.get("/wiki/pages", response_model=WikiPagesResponse, tags=["wiki"])
def list_wiki_pages(service: Any = Depends(get_service)) -> WikiPagesResponse:
    try:
        result = service.list_wiki_pages()
        values: Iterable[Any]
        if isinstance(result, Mapping):
            values = result.get("pages", [])
        else:
            values = result

        pages = []
        for value in values:
            if isinstance(value, str):
                pages.append(WikiPageResponse(relative_path=value))
            else:
                item = model_to_dict(value)
                pages.append(
                    WikiPageResponse(
                        relative_path=str(item.get("relative_path", item.get("path", ""))),
                        title=item.get("title"),
                        summary=item.get("summary"),
                        source_paths=[str(path) for path in item.get("source_paths", [])],
                        size_bytes=item.get("size_bytes"),
                        modified_at=item.get("modified_at"),
                    )
                )
        return WikiPagesResponse(pages=pages)
    except Exception as exc:
        raise _service_error("list wiki pages", exc) from exc


@app.get("/wiki/lint", response_model=WikiLintResponse, tags=["wiki"])
def lint_wiki(service: Any = Depends(get_service)) -> WikiLintResponse:
    try:
        payload = model_to_dict(service.lint_wiki())
        return WikiLintResponse(
            valid=bool(payload.get("valid", False)),
            pages_checked=int(payload.get("pages_checked", 0)),
            issues=[WikiLintIssue(**model_to_dict(issue)) for issue in payload.get("issues", [])],
            graph=WikiGraphSummaryResponse(
                **model_to_dict(
                    payload.get(
                        "graph",
                        {
                            "pages": 0,
                            "links": 0,
                            "without_incoming": 0,
                            "without_outgoing": 0,
                            "isolated": 0,
                        },
                    )
                )
            ),
        )
    except Exception as exc:
        raise _service_error("lint the wiki", exc) from exc


@app.post(
    "/wiki/lint/repair-links",
    response_model=LinkRepairResponse,
    tags=["wiki"],
)
def repair_wiki_links(
    request: LinkRepairRequest,
    service: Any = Depends(get_service),
) -> LinkRepairResponse:
    try:
        payload = model_to_dict(service.repair_wiki_links(max_links=request.max_links))
        return LinkRepairResponse(
            links_added=[
                LinkProposalResponse(**model_to_dict(item))
                for item in payload.get("links_added", [])
            ],
            pages_updated=[str(path) for path in payload.get("pages_updated", [])],
            graph_before=WikiGraphSummaryResponse(
                **model_to_dict(payload.get("graph_before", {}))
            ),
            graph_after=WikiGraphSummaryResponse(
                **model_to_dict(payload.get("graph_after", {}))
            ),
            usage={str(key): int(value) for key, value in payload.get("usage", {}).items()},
            warning=payload.get("warning"),
        )
    except Exception as exc:
        raise _service_error("repair wiki links", exc) from exc
