from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .models import RagChunk, SearchHit


class VectorStore(Protocol):
    def ensure_collection(self) -> None: ...
    def upsert_document(
        self, document_id: str, chunks: Sequence[RagChunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...
    def count_document(self, document_id: str) -> int: ...
    def delete_document(self, document_id: str) -> None: ...
    def search(
        self, vector: Sequence[float], *, limit: int, document_ids: list[str] | None = None
    ) -> list[SearchHit]: ...
    def health(self) -> bool: ...


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        dimension: int,
        client: Any | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is not installed.") from exc
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = client or QdrantClient(url=url.rstrip("/"), timeout=30)

    @staticmethod
    def _models() -> Any:
        from qdrant_client import models

        return models

    def ensure_collection(self) -> None:
        models = self._models()
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )
            for field in ("document_id", "source_hash", "filename"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            return
        info = self.client.get_collection(self.collection_name)
        size = getattr(info.config.params.vectors, "size", None)
        if size is not None and int(size) != self.dimension:
            raise RuntimeError(
                f"Collection {self.collection_name} has dimension {size}; expected {self.dimension}."
            )

    def upsert_document(
        self,
        document_id: str,
        chunks: Sequence[RagChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be non-empty and have equal length.")
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("At least one vector has the wrong dimension.")
        self.ensure_collection()
        self.delete_document(document_id)
        models = self._models()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload=self._payload(chunk),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    def delete_document(self, document_id: str) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        models = self._models()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def count_document(self, document_id: str) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        models = self._models()
        return int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            ).count
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection_name):
            return []
        if len(vector) != self.dimension:
            raise ValueError("Query vector has the wrong dimension.")
        models = self._models()
        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=value)
                    )
                    for value in document_ids
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._hit(point) for point in response.points]

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        # Health probes must translate every client/transport failure into a
        # degraded response; the actual operation paths still raise errors.
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _payload(chunk: RagChunk) -> dict[str, object]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_hash": chunk.source_hash,
            "filename": chunk.filename,
            "title": chunk.title,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "embedding_text": chunk.embedding_text,
            "token_count": chunk.token_count,
            "element_ids": chunk.element_ids,
            "page_numbers": chunk.page_numbers,
            "heading_path": chunk.heading_path,
            "content_types": [value.value for value in chunk.content_types],
        }

    @staticmethod
    def _hit(point: Any) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
            chunk_id=str(payload.get("chunk_id", point.id)),
            document_id=str(payload.get("document_id", "")),
            filename=str(payload.get("filename", "")),
            title=str(payload.get("title", "")),
            score=float(point.score),
            text=str(payload.get("text", "")),
            page_numbers=[int(value) for value in payload.get("page_numbers", [])],
            heading_path=[str(value) for value in payload.get("heading_path", [])],
            content_types=[str(value) for value in payload.get("content_types", [])],
            metadata={
                "source_hash": str(payload.get("source_hash", "")),
                "ordinal": int(payload.get("ordinal", 0)),
                "token_count": int(payload.get("token_count", 0)),
                "element_ids": [str(value) for value in payload.get("element_ids", [])],
            },
        )
