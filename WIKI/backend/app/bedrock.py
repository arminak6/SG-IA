"""A small, model-agnostic wrapper around Bedrock Runtime Converse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import BedrockSettings


class BedrockError(RuntimeError):
    """Raised for malformed responses or unavailable Bedrock dependencies."""


@dataclass(frozen=True)
class ConverseTurn:
    message: dict[str, Any]
    stop_reason: str
    usage: dict[str, int]
    metrics: dict[str, Any]


class BedrockConverseClient:
    """Invoke Bedrock Runtime through its normalized Converse API.

    Passing ``client`` is useful for unit tests.  When omitted, Boto3 is loaded
    only on the first request, so importing the application does not require
    either Boto3 or live AWS credentials.
    """

    def __init__(self, settings: BedrockSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def _make_client(self) -> Any:
        try:
            import boto3  # type: ignore
            from botocore.config import Config  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise BedrockError("Boto3 is required to call AWS Bedrock.") from exc

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
                read_timeout=60,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
            **client_kwargs,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def converse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        tool_choice: Mapping[str, Any] | None = None,
    ) -> ConverseTurn:
        request: dict[str, Any] = {
            "modelId": self.settings.bedrock_model_id,
            "messages": [dict(message) for message in messages],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": max_tokens or self.settings.max_output_tokens,
                "temperature": temperature,
            },
        }
        if tools:
            tool_config: dict[str, Any] = {"tools": [dict(tool) for tool in tools]}
            if tool_choice:
                tool_config["toolChoice"] = dict(tool_choice)
            request["toolConfig"] = tool_config

        for attempt in range(2):
            try:
                response = self.client.converse(**request)
                break
            except Exception as exc:
                # Keep the original exception chained for diagnostics, but never add
                # settings or request data (which may contain private documents).
                error_response = getattr(exc, "response", None)
                error = (
                    error_response.get("Error", {})
                    if isinstance(error_response, Mapping)
                    else {}
                )
                error_code = error.get("Code") if isinstance(error, Mapping) else None
                # Bedrock occasionally rejects an otherwise unchanged valid request
                # transiently. A validation rejection happens before model output, so
                # one immediate retry cannot duplicate a tool action or wiki write.
                if error_code == "ValidationException" and attempt == 0:
                    continue
                code_suffix = f" ({error_code})" if isinstance(error_code, str) else ""
                raise BedrockError(
                    f"Bedrock Converse request failed: {type(exc).__name__}{code_suffix}."
                ) from exc

        try:
            message = response["output"]["message"]
            role = message["role"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise BedrockError("Bedrock returned an invalid Converse response.") from exc
        if role != "assistant" or not isinstance(content, list):
            raise BedrockError("Bedrock returned an invalid assistant message.")

        usage = response.get("usage", {})
        metrics = response.get("metrics", {})
        return ConverseTurn(
            message={"role": role, "content": content},
            stop_reason=str(response.get("stopReason", "unknown")),
            usage={str(key): int(value) for key, value in usage.items() if isinstance(value, int)},
            metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
        )
