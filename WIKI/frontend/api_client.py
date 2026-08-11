"""Small, typed HTTP client for the LLM Wiki backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


class WikiApiError(RuntimeError):
    """Raised when the backend cannot complete a request."""


class WikiApiUnavailable(WikiApiError):
    """Raised when the backend cannot be reached."""


@dataclass(frozen=True)
class ApiDocument:
    relative_path: str
    status: str
    size_bytes: int
    modified_at: str | float | int | None = None

    @property
    def is_ingested(self) -> bool:
        return self.status.casefold() in {"ingested", "ready"}

    @property
    def modified_timestamp(self) -> float:
        value = self.modified_at
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value:
            try:
                normalized = value.replace("Z", "+00:00")
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                pass
        return datetime.now(tz=timezone.utc).timestamp()


@dataclass(frozen=True)
class FailedUpdate:
    path: str
    error: str


@dataclass(frozen=True)
class ProcessedUpdate:
    path: str
    message: str = ""


@dataclass(frozen=True)
class SkippedUpdate:
    path: str
    reason: str = ""


@dataclass(frozen=True)
class UpdateResult:
    processed: tuple[ProcessedUpdate, ...]
    skipped: tuple[SkippedUpdate, ...]
    failed: tuple[FailedUpdate, ...]

    def as_dict(self) -> dict[str, list[Any]]:
        """Return a session-state-friendly representation."""

        return {
            "processed": [
                {"path": item.path, "message": item.message}
                for item in self.processed
            ],
            "skipped": [
                {"path": item.path, "reason": item.reason} for item in self.skipped
            ],
            "failed": [
                {"path": item.path, "error": item.error} for item in self.failed
            ],
        }


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    citations: tuple[str, ...]
    confidence_score: float | None = None


class WikiApiClient:
    """Access the FastAPI service without leaking transport details into the UI."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        configured_url = base_url or os.getenv("LLM_WIKI_API_URL", DEFAULT_API_URL)
        self.base_url = configured_url.rstrip("/")
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=(1.0, 15.0))

    def list_documents(self) -> list[ApiDocument]:
        payload = self._request("GET", "/documents", timeout=(1.0, 10.0))
        items = _extract_list(payload, "documents")
        documents: list[ApiDocument] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise WikiApiError("The backend returned an invalid document entry.")
            relative_path = str(item.get("relative_path", "")).replace("\\", "/")
            if not relative_path:
                raise WikiApiError("The backend returned a document without a path.")
            documents.append(
                ApiDocument(
                    relative_path=relative_path,
                    status=str(item.get("status", "Pending")).title(),
                    size_bytes=_safe_int(item.get("size_bytes")),
                    modified_at=item.get("modified_at"),
                )
            )
        return documents

    def list_pages(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/wiki/pages", timeout=(1.0, 10.0))
        items = _extract_list(payload, "pages")
        if not all(isinstance(item, Mapping) for item in items):
            raise WikiApiError("The backend returned an invalid wiki page list.")
        return [dict(item) for item in items]

    def update_wiki(self, paths: Sequence[str] | None = None) -> UpdateResult:
        body: dict[str, Any] = {}
        if paths is not None:
            body["paths"] = list(paths)
        payload = self._request(
            "POST",
            "/wiki/update",
            json=body,
            timeout=(3.05, 600.0),
        )
        if not isinstance(payload, Mapping):
            raise WikiApiError("The backend returned an invalid update result.")

        processed: list[ProcessedUpdate] = []
        for item in _as_list(payload.get("processed", [])):
            if isinstance(item, Mapping):
                processed.append(
                    ProcessedUpdate(
                        path=_item_path(item),
                        message=str(item.get("message", "")),
                    )
                )
            else:
                processed.append(ProcessedUpdate(path=str(item)))

        skipped: list[SkippedUpdate] = []
        for item in _as_list(payload.get("skipped", [])):
            if isinstance(item, Mapping):
                skipped.append(
                    SkippedUpdate(
                        path=_item_path(item),
                        reason=str(item.get("reason", "")),
                    )
                )
            else:
                skipped.append(SkippedUpdate(path=str(item)))

        failed: list[FailedUpdate] = []
        for item in _as_list(payload.get("failed", [])):
            if isinstance(item, Mapping):
                failed.append(
                    FailedUpdate(
                        path=_item_path(item),
                        error=str(item.get("error", "Unknown error")),
                    )
                )
            else:
                failed.append(FailedUpdate(path=str(item), error="Update failed"))

        return UpdateResult(
            processed=tuple(processed),
            skipped=tuple(skipped),
            failed=tuple(failed),
        )

    def chat(self, question: str) -> ChatResponse:
        payload = self._request(
            "POST",
            "/chat",
            json={"question": question},
            timeout=(3.05, 180.0),
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("answer"), str):
            raise WikiApiError("The backend returned an invalid chat response.")
        citations = tuple(
            _format_citation(item) for item in _as_list(payload.get("citations", []))
        )
        raw_confidence = payload.get("confidence_score")
        confidence_score: float | None = None
        if raw_confidence is not None:
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
                raise WikiApiError("The backend returned an invalid confidence score.")
            confidence_score = float(raw_confidence)
            if not 0 <= confidence_score <= 10:
                raise WikiApiError("The backend returned an invalid confidence score.")
        return ChatResponse(
            answer=payload["answer"],
            citations=citations,
            confidence_score=confidence_score,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | tuple[float, float],
        **kwargs: Any,
    ) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise WikiApiUnavailable(
                f"Cannot reach the LLM Wiki API at {self.base_url}."
            ) from exc

        if not response.ok:
            detail = _response_error_detail(response)
            raise WikiApiError(f"Backend request failed ({response.status_code}): {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise WikiApiError("The backend returned invalid JSON.") from exc


def _extract_list(payload: Any, envelope_key: str) -> list[Any]:
    if isinstance(payload, Mapping):
        value = payload.get(envelope_key, [])
    else:
        value = payload
    if not isinstance(value, list):
        raise WikiApiError(f"The backend returned an invalid {envelope_key} list.")
    return value


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_citation(value: Any) -> str:
    if isinstance(value, Mapping):
        label = ""
        for key in ("wiki_path", "path", "source", "relative_path", "title"):
            if value.get(key):
                label = str(value[key])
                break
        sources = [str(item) for item in _as_list(value.get("source_paths", []))]
        if sources:
            source_label = ", ".join(sources)
            return f"{label} (sources: {source_label})" if label else source_label
        if label:
            return label
    return str(value)


def _item_path(item: Mapping[str, Any]) -> str:
    return str(item.get("source_path") or item.get("path") or "Unknown document")


def _response_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "Unknown error"
    if isinstance(payload, Mapping):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return str(detail)
    return response.text.strip() or "Unknown error"
