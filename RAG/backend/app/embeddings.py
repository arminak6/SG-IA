from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def build_boto3_session(*, region: str, credentials_file: Path | None = None) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed.") from exc
    if credentials_file is None:
        return boto3.Session(region_name=region)
    try:
        payload = json.loads(credentials_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The configured AWS credentials file is unreadable.") from exc
    return boto3.Session(
        aws_access_key_id=payload.get("aws_access_key_id"),
        aws_secret_access_key=payload.get("aws_secret_access_key"),
        aws_session_token=payload.get("aws_session_token"),
        region_name=payload.get("region") or region,
        profile_name=payload.get("profile_name"),
    )


class BedrockTitanEmbeddingProvider:
    def __init__(
        self,
        *,
        session: Any,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 512,
        client: Any | None = None,
    ):
        if dimension not in {256, 512, 1024}:
            raise ValueError("Titan V2 supports 256, 512, or 1024 dimensions.")
        self.session = session
        self.model_id = model_id
        self.dimension = dimension
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError("botocore is not installed.") from exc
            self._client = self.session.client(
                "bedrock-runtime",
                config=Config(
                    retries={"max_attempts": 6, "mode": "adaptive"},
                    connect_timeout=10,
                    read_timeout=60,
                ),
            )
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot embed blank text.")
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "inputText": cleaned[:50_000],
                    "dimensions": self.dimension,
                    "normalize": True,
                    "embeddingTypes": ["float"],
                }
            ),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        vector = payload.get("embedding")
        if vector is None:
            vector = payload.get("embeddingsByType", {}).get("float")
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise RuntimeError("Titan returned an embedding with the wrong dimension.")
        return [float(value) for value in vector]

