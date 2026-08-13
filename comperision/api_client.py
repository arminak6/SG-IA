"""Concurrent HTTP client for comparing the RAG and LLM Wiki backends."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import requests


DEFAULT_RAG_API_URL = "http://127.0.0.1:8001"
DEFAULT_WIKI_API_URL = "http://127.0.0.1:8002"
DEFAULT_CHAT_TIMEOUT_SECONDS = 660.0


class ComparisonApiError(RuntimeError):
    """Raised when a comparison backend cannot return a valid response."""


@dataclass(frozen=True)
class BackendAnswer:
    """Normalized view of one backend's native chat response."""

    approach: str
    status: str = "error"
    answer: str = ""
    citations: tuple[dict[str, Any], ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    server_latency_ms: float | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    embedding_model_id: str | None = None
    confidence_score: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    manager_action: dict[str, Any] | None = None
    correction: dict[str, Any] | None = None
    client_elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [dict(item) for item in self.citations]
        return payload


@dataclass(frozen=True)
class BackendHealth:
    approach: str
    healthy: bool
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    client_elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackendClient:
    """Small transport wrapper; all knowledge and reasoning stay in the APIs."""

    def __init__(
        self,
        approach: str,
        base_url: str,
        *,
        session: requests.Session | None = None,
        chat_timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
    ) -> None:
        if approach not in {"rag", "wiki"}:
            raise ValueError(f"Unsupported approach: {approach}")
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError(f"The {approach.upper()} API URL cannot be empty.")

        self.approach = approach
        self.base_url = normalized_url
        self.session = session or requests.Session()
        self.chat_timeout_seconds = chat_timeout_seconds

    def health(self) -> BackendHealth:
        started = time.perf_counter()
        try:
            payload = self._request("GET", "/health", timeout=(2.0, 15.0))
            if not isinstance(payload, Mapping):
                raise ComparisonApiError("The backend returned an invalid health response.")
            status = str(payload.get("status", "unknown"))
            return BackendHealth(
                approach=self.approach,
                healthy=status.casefold() in {"ok", "healthy", "ready"},
                status=status,
                details=dict(payload),
                client_elapsed_ms=_elapsed_ms(started),
            )
        except (ComparisonApiError, requests.RequestException) as exc:
            return BackendHealth(
                approach=self.approach,
                healthy=False,
                status="unavailable",
                client_elapsed_ms=_elapsed_ms(started),
                error=_friendly_error(exc),
            )

    def chat(
        self,
        question: str,
        *,
        session_id: str,
        rag_top_k: int = 10,
    ) -> BackendAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be blank.")

        body: dict[str, Any] = {
            "question": normalized_question,
            "session_id": session_id,
        }
        if self.approach == "rag":
            body["top_k"] = rag_top_k

        started = time.perf_counter()
        payload = self._request(
            "POST",
            "/chat",
            json=body,
            timeout=(5.0, self.chat_timeout_seconds),
        )
        return _normalize_answer(
            self.approach,
            payload,
            client_elapsed_ms=_elapsed_ms(started),
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
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = _response_detail(exc.response)
            suffix = f": {detail}" if detail else ""
            raise ComparisonApiError(
                f"{self.approach.upper()} returned HTTP "
                f"{exc.response.status_code}{suffix}"
            ) from exc
        except requests.RequestException as exc:
            raise ComparisonApiError(
                f"Could not reach {self.approach.upper()} at {self.base_url}."
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ComparisonApiError(
                f"{self.approach.upper()} returned a non-JSON response."
            ) from exc


def ask_both(
    question: str,
    *,
    session_id: str,
    rag_api_url: str | None = None,
    wiki_api_url: str | None = None,
    rag_top_k: int = 10,
    chat_timeout_seconds: float | None = None,
    clients: Mapping[str, BackendClient] | None = None,
) -> dict[str, BackendAnswer]:
    """Send exactly the same question to both APIs concurrently.

    A failed backend is converted to an error result so the successful answer
    remains visible. ``clients`` is injectable for deterministic offline tests.
    """

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question cannot be blank.")

    if clients is None:
        timeout = chat_timeout_seconds or _configured_chat_timeout()
        clients = {
            "wiki": BackendClient(
                "wiki",
                wiki_api_url or os.getenv("WIKI_API_URL", DEFAULT_WIKI_API_URL),
                chat_timeout_seconds=timeout,
            ),
            "rag": BackendClient(
                "rag",
                rag_api_url or os.getenv("RAG_API_URL", DEFAULT_RAG_API_URL),
                chat_timeout_seconds=timeout,
            ),
        }

    required = {"wiki", "rag"}
    if set(clients) != required:
        raise ValueError("Comparison requires exactly one 'wiki' and one 'rag' client.")

    answers: dict[str, BackendAnswer] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="comparison-chat") as pool:
        futures = {
            pool.submit(
                _safe_chat,
                client,
                normalized_question,
                session_id,
                rag_top_k,
            ): approach
            for approach, client in clients.items()
        }
        for future in as_completed(futures):
            approach = futures[future]
            answers[approach] = future.result()

    return {"wiki": answers["wiki"], "rag": answers["rag"]}


def check_both(
    *,
    rag_api_url: str | None = None,
    wiki_api_url: str | None = None,
    clients: Mapping[str, BackendClient] | None = None,
) -> dict[str, BackendHealth]:
    """Check both API health endpoints concurrently."""

    if clients is None:
        clients = {
            "wiki": BackendClient(
                "wiki", wiki_api_url or os.getenv("WIKI_API_URL", DEFAULT_WIKI_API_URL)
            ),
            "rag": BackendClient(
                "rag", rag_api_url or os.getenv("RAG_API_URL", DEFAULT_RAG_API_URL)
            ),
        }

    health: dict[str, BackendHealth] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="comparison-health") as pool:
        futures = {pool.submit(client.health): approach for approach, client in clients.items()}
        for future in as_completed(futures):
            approach = futures[future]
            health[approach] = future.result()
    return {"wiki": health["wiki"], "rag": health["rag"]}


def _safe_chat(
    client: BackendClient,
    question: str,
    session_id: str,
    rag_top_k: int,
) -> BackendAnswer:
    started = time.perf_counter()
    try:
        return client.chat(
            question,
            session_id=session_id,
            rag_top_k=rag_top_k,
        )
    except Exception as exc:  # Each backend must fail independently.
        return BackendAnswer(
            approach=client.approach,
            client_elapsed_ms=_elapsed_ms(started),
            error=_friendly_error(exc),
        )


def _normalize_answer(
    approach: str,
    payload: Any,
    *,
    client_elapsed_ms: float,
) -> BackendAnswer:
    if not isinstance(payload, Mapping):
        raise ComparisonApiError(f"{approach.upper()} returned an invalid chat response.")
    if not isinstance(payload.get("answer"), str):
        raise ComparisonApiError(f"{approach.upper()} returned no answer text.")

    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list) or not all(
        isinstance(item, Mapping) for item in raw_citations
    ):
        raise ComparisonApiError(f"{approach.upper()} returned invalid citations.")

    confidence = payload.get("confidence_score")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ComparisonApiError(
                f"{approach.upper()} returned an invalid confidence score."
            )
        confidence = float(confidence)

    server_latency = payload.get("latency_ms")
    if isinstance(server_latency, bool) or not isinstance(server_latency, (int, float)):
        server_latency = None

    return BackendAnswer(
        approach=approach,
        status=str(payload.get("status", "answered")),
        answer=str(payload["answer"]),
        citations=tuple(dict(item) for item in raw_citations),
        usage=_mapping(payload.get("usage")),
        server_latency_ms=float(server_latency) if server_latency is not None else None,
        timings=_mapping(payload.get("timings")),
        model_id=_optional_string(payload.get("model_id")),
        embedding_model_id=_optional_string(payload.get("embedding_model_id")),
        confidence_score=confidence,
        debug=_mapping(payload.get("debug")),
        manager_action=_optional_mapping(payload.get("manager_action")),
        correction=_optional_mapping(payload.get("correction")),
        client_elapsed_ms=client_elapsed_ms,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _response_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
    except ValueError:
        return ""
    if isinstance(payload, Mapping):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail[:500]
    return ""


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, (ComparisonApiError, ValueError)):
        return str(exc)
    return f"The backend request failed ({type(exc).__name__})."


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _configured_chat_timeout() -> float:
    raw_value = os.getenv(
        "COMPARISON_CHAT_TIMEOUT_SECONDS", str(DEFAULT_CHAT_TIMEOUT_SECONDS)
    )
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_CHAT_TIMEOUT_SECONDS
    return max(30.0, value)
