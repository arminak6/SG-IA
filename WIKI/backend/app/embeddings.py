"""Configurable Amazon Bedrock embeddings for semantic Wiki search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .config import BedrockSettings


class EmbeddingError(RuntimeError):
    """Raised when an embedding request or response is unusable."""


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    input_tokens: int = 0


class TextEmbedder(Protocol):
    model_id: str
    dimensions: int
    max_input_characters: int

    def embed(self, text: str) -> EmbeddingResult: ...


class TitanEmbeddingClient:
    """Invoke Amazon Titan Text Embeddings V2 through Bedrock Runtime.

    The client is lazy so importing and testing the application never requires
    live AWS access. A scripted ``client`` can be injected by offline tests.
    """

    def __init__(self, settings: BedrockSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.model_id = settings.embedding_model_id
        self.dimensions = settings.embedding_dimensions
        self.max_input_characters = settings.embedding_max_input_characters
        self._client = client

    def _make_client(self) -> Any:
        try:
            import boto3  # type: ignore
            from botocore.config import Config  # type: ignore
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise EmbeddingError("Boto3 is required for Bedrock semantic search.") from exc

        session_kwargs: dict[str, str] = {}
        if self.settings.aws_access_key_id:
            session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key or ""
            if self.settings.aws_session_token:
                session_kwargs["aws_session_token"] = self.settings.aws_session_token

        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, str] = {}
        if self.settings.region_name:
            client_kwargs["region_name"] = self.settings.region_name
        return session.client(
            "bedrock-runtime",
            config=Config(
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            **client_kwargs,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def embed(self, text: str) -> EmbeddingResult:
        normalized = str(text).replace("\x00", " ").strip()
        if not normalized:
            raise EmbeddingError("Embedding input cannot be empty.")
        bounded = normalized[: self.max_input_characters]
        request_body = json.dumps(
            {
                "inputText": bounded,
                "dimensions": self.dimensions,
                "normalize": True,
            }
        )
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=request_body,
                accept="application/json",
                contentType="application/json",
            )
            response_body = response.get("body")
            raw_payload = response_body.read() if hasattr(response_body, "read") else response_body
            if isinstance(raw_payload, bytes):
                raw_payload = raw_payload.decode("utf-8")
            payload = json.loads(raw_payload)
        except Exception as exc:
            error_response = getattr(exc, "response", None)
            error = (
                error_response.get("Error", {})
                if isinstance(error_response, Mapping)
                else {}
            )
            error_code = error.get("Code") if isinstance(error, Mapping) else None
            suffix = f" ({error_code})" if isinstance(error_code, str) else ""
            raise EmbeddingError(
                f"Bedrock embedding request failed: {type(exc).__name__}{suffix}."
            ) from exc

        vector = payload.get("embedding") if isinstance(payload, Mapping) else None
        if not isinstance(vector, list) or len(vector) != self.dimensions:
            raise EmbeddingError("Bedrock returned an invalid embedding vector.")
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Bedrock returned a non-numeric embedding vector.") from exc
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError("Bedrock returned a non-finite embedding vector.")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise EmbeddingError("Bedrock returned an empty embedding direction.")
        # Normalize defensively even though Titan is requested to normalize.
        normalized_vector = tuple(value / norm for value in values)
        token_count = payload.get("inputTextTokenCount", 0)
        return EmbeddingResult(
            vector=normalized_vector,
            input_tokens=int(token_count) if isinstance(token_count, int) else 0,
        )
