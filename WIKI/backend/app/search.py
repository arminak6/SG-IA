"""Hybrid lexical and semantic search over generated Wiki pages."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from filelock import FileLock

from .embeddings import EmbeddingError, TextEmbedder
from .repository import RepositoryError, WikiRepository, WikiSearchResult


logger = logging.getLogger(__name__)


class WikiSearchError(RuntimeError):
    """Raised when the semantic index cannot be built or read safely."""


@dataclass(frozen=True)
class SearchResponse:
    results: tuple[WikiSearchResult, ...]
    mode: str
    embedding_input_tokens: int = 0


@dataclass(frozen=True)
class IndexRefreshResult:
    pages_total: int
    pages_embedded: int
    pages_cached: int
    pages_removed: int
    embedding_input_tokens: int


class HybridWikiSearch:
    """Combine exact keyword ranking with cached vector similarity.

    Embeddings represent generated Wiki pages only. The answer agent must still
    read complete pages before it may cite or use them as evidence.
    """

    CACHE_VERSION = 1
    CACHE_FILENAME = ".semantic-index.json"

    def __init__(
        self,
        repository: WikiRepository,
        embedder: TextEmbedder | None,
        *,
        cache_path: Path | None = None,
        lexical_weight: float = 0.55,
        semantic_weight: float = 0.45,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.cache_path = cache_path or repository.wiki_root / self.CACHE_FILENAME
        self.lexical_weight = float(lexical_weight)
        self.semantic_weight = float(semantic_weight)
        self._refresh_lock = threading.Lock()
        self._file_lock = FileLock(str(self.cache_path) + ".lock")

    @property
    def enabled(self) -> bool:
        return self.embedder is not None

    def _empty_cache(self) -> dict[str, object]:
        if self.embedder is None:
            return {"version": self.CACHE_VERSION, "pages": {}}
        return {
            "version": self.CACHE_VERSION,
            "model_id": self.embedder.model_id,
            "dimensions": self.embedder.dimensions,
            "pages": {},
        }

    def _load_cache(self) -> dict[str, object]:
        if self.embedder is None or not self.cache_path.is_file():
            return self._empty_cache()
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_cache()
        if not isinstance(payload, dict):
            return self._empty_cache()
        if (
            payload.get("version") != self.CACHE_VERSION
            or payload.get("model_id") != self.embedder.model_id
            or payload.get("dimensions") != self.embedder.dimensions
            or not isinstance(payload.get("pages"), dict)
        ):
            return self._empty_cache()
        return payload

    def _write_cache(self, payload: Mapping[str, object]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.",
            suffix=".tmp",
            dir=str(self.cache_path.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _page_digest(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _document_text(path: str, title: str, summary: str, content: str) -> str:
        return f"Wiki path: {path}\nTitle: {title}\nSummary: {summary}\n\n{content}"

    @staticmethod
    def _valid_vector(value: object, dimensions: int) -> tuple[float, ...] | None:
        if not isinstance(value, list) or len(value) != dimensions:
            return None
        try:
            vector = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in vector):
            return None
        norm = math.sqrt(sum(item * item for item in vector))
        if norm <= 0:
            return None
        return tuple(item / norm for item in vector)

    def refresh(self) -> IndexRefreshResult:
        """Embed new or changed Wiki pages and remove deleted cache entries."""

        if self.embedder is None:
            pages_total = len(self.repository.list_wiki_pages())
            return IndexRefreshResult(pages_total, 0, pages_total, 0, 0)

        with self._refresh_lock, self._file_lock:
            cache = self._load_cache()
            cached_pages = dict(cache.get("pages", {}))
            pages = self.repository.list_wiki_pages()
            known_paths = {page.path for page in pages}
            removed = len(set(cached_pages) - known_paths)
            changed = removed > 0
            input_tokens = 0
            embedded = 0
            retained: dict[str, object] = {}

            for page in pages:
                content = self.repository.read_wiki_page(page.path)
                digest = self._page_digest(content)
                cached = cached_pages.get(page.path)
                if isinstance(cached, Mapping) and cached.get("sha256") == digest:
                    vector = self._valid_vector(
                        cached.get("embedding"), self.embedder.dimensions
                    )
                    if vector is not None:
                        retained[page.path] = dict(cached)
                        continue

                result = self.embedder.embed(
                    self._document_text(page.path, page.title, page.summary, content)
                )
                retained[page.path] = {
                    "sha256": digest,
                    "title": page.title,
                    "summary": page.summary,
                    "embedding": list(result.vector),
                }
                embedded += 1
                input_tokens += result.input_tokens
                changed = True

            if changed or not self.cache_path.is_file():
                cache = {
                    "version": self.CACHE_VERSION,
                    "model_id": self.embedder.model_id,
                    "dimensions": self.embedder.dimensions,
                    "pages": retained,
                }
                try:
                    self._write_cache(cache)
                except OSError as exc:
                    raise WikiSearchError("Could not persist the semantic Wiki index.") from exc

            return IndexRefreshResult(
                pages_total=len(pages),
                pages_embedded=embedded,
                pages_cached=len(pages) - embedded,
                pages_removed=removed,
                embedding_input_tokens=input_tokens,
            )

    def _semantic_results(
        self, query: str, *, limit: int
    ) -> tuple[list[WikiSearchResult], int]:
        if self.embedder is None:
            return [], 0
        refresh = self.refresh()
        with self._file_lock:
            cache = self._load_cache()
        cached_pages = cache.get("pages", {})
        if not isinstance(cached_pages, Mapping) or not cached_pages:
            return [], refresh.embedding_input_tokens

        query_result = self.embedder.embed(query)
        matches: list[WikiSearchResult] = []
        for path, value in cached_pages.items():
            if not isinstance(path, str) or not isinstance(value, Mapping):
                continue
            vector = self._valid_vector(value.get("embedding"), self.embedder.dimensions)
            if vector is None:
                continue
            similarity = sum(
                left * right for left, right in zip(query_result.vector, vector)
            )
            matches.append(
                WikiSearchResult(
                    path=path,
                    title=str(value.get("title", path)),
                    excerpt=str(value.get("summary", "")),
                    score=round(similarity, 6),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.path.casefold()))
        return matches[: max(1, min(int(limit), 20))], (
            refresh.embedding_input_tokens + query_result.input_tokens
        )

    def _combine(
        self,
        lexical: list[WikiSearchResult],
        semantic: list[WikiSearchResult],
        *,
        limit: int,
    ) -> list[WikiSearchResult]:
        # Weighted reciprocal-rank fusion avoids comparing unrelated lexical
        # count and cosine-similarity scales.
        scores: dict[str, float] = {}
        items: dict[str, WikiSearchResult] = {}
        for rank, item in enumerate(lexical, start=1):
            scores[item.path] = scores.get(item.path, 0.0) + self.lexical_weight / (60 + rank)
            items[item.path] = item
        for rank, item in enumerate(semantic, start=1):
            scores[item.path] = scores.get(item.path, 0.0) + self.semantic_weight / (60 + rank)
            # Prefer lexical excerpts when both exist because they show exact matches.
            items.setdefault(item.path, item)
        ranked = sorted(scores, key=lambda path: (-scores[path], path.casefold()))
        return [
            WikiSearchResult(
                path=path,
                title=items[path].title,
                excerpt=items[path].excerpt,
                score=round(scores[path] * 10_000, 6),
            )
            for path in ranked[: max(1, min(int(limit), 20))]
        ]

    def search(self, query: str, *, limit: int = 8) -> SearchResponse:
        bounded_limit = max(1, min(int(limit), 20))
        candidate_limit = min(20, max(bounded_limit * 3, bounded_limit))
        lexical = self.repository.search_wiki(query, limit=candidate_limit)
        if self.embedder is None:
            return SearchResponse(tuple(lexical[:bounded_limit]), "lexical")
        try:
            semantic, input_tokens = self._semantic_results(
                query, limit=candidate_limit
            )
            combined = self._combine(lexical, semantic, limit=bounded_limit)
            return SearchResponse(tuple(combined), "hybrid", input_tokens)
        except (EmbeddingError, RepositoryError, WikiSearchError, OSError, ValueError) as exc:
            # Searching must remain available during embedding outages. Log only
            # the safe exception type and use deterministic lexical results.
            logger.warning(
                "Semantic Wiki search unavailable; using lexical fallback (%s).",
                type(exc).__name__,
            )
            return SearchResponse(tuple(lexical[:bounded_limit]), "lexical_fallback")
