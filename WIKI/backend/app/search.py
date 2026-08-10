"""Section-level hybrid search over generated Wiki pages."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from filelock import FileLock

from .embeddings import EmbeddingError, TextEmbedder
from .repository import RepositoryError, WikiRepository, WikiSearchResult


logger = logging.getLogger(__name__)


class WikiSearchError(RuntimeError):
    """Raised when the semantic index cannot be built or read safely."""


@dataclass(frozen=True)
class SectionMatch:
    """A semantic hit inside a parent Wiki page."""

    section_id: str
    heading: str
    heading_path: str
    excerpt: str
    semantic_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "excerpt": self.excerpt,
            "semantic_score": self.semantic_score,
        }


@dataclass(frozen=True)
class PageSearchDiagnostic:
    """Explain how a parent page reached the final candidate ranking."""

    path: str
    title: str
    final_rank: int
    fused_score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    matched_sections: tuple[SectionMatch, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "title": self.title,
            "final_rank": self.final_rank,
            "fused_score": self.fused_score,
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
            "matched_sections": [match.to_dict() for match in self.matched_sections],
        }


@dataclass(frozen=True)
class SearchResponse:
    results: tuple[WikiSearchResult, ...]
    mode: str
    embedding_input_tokens: int = 0
    diagnostics: tuple[PageSearchDiagnostic, ...] = ()


@dataclass(frozen=True)
class IndexRefreshResult:
    pages_total: int
    pages_embedded: int
    pages_cached: int
    pages_removed: int
    embedding_input_tokens: int
    sections_total: int = 0
    sections_embedded: int = 0
    sections_cached: int = 0
    sections_removed: int = 0


@dataclass(frozen=True)
class _WikiSection:
    section_id: str
    heading: str
    heading_path: str
    level: int
    text: str
    sha256: str


class HybridWikiSearch:
    """Fuse page-level lexical rank with section-level vector similarity.

    Embeddings are navigation aids over generated Wiki sections. Semantic hits
    are aggregated to unique parent pages, and the answer agent must still read
    each complete parent page before it can use or cite that page as evidence.
    """

    CACHE_VERSION = 2
    CACHE_FILENAME = ".semantic-index.json"
    HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
    FRONTMATTER_PATTERN = re.compile(
        r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
    )
    # These headings conventionally contain provenance/navigation rather than
    # answer evidence. The rule is structural and corpus-independent.
    PROVENANCE_HEADINGS = frozenset(
        {"source", "sources", "reference", "references", "provenance", "fonti"}
    )
    MAX_MATCHED_SECTIONS_PER_PAGE = 3

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
    def _digest(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_heading(value: str) -> str:
        # Keep heading text readable while removing lightweight Markdown syntax.
        value = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", value)
        value = re.sub(r"[`*_~]", "", value)
        return " ".join(value.split()).strip()

    @classmethod
    def _sections(cls, content: str, *, fallback_title: str) -> list[_WikiSection]:
        """Split Markdown into heading blocks while preserving heading ancestry."""

        body = cls.FRONTMATTER_PATTERN.sub("", content, count=1)
        raw_sections: list[tuple[int, str, str, list[str]]] = []
        heading_stack: list[tuple[int, str]] = []
        current_level = 0
        current_heading = fallback_title.strip() or "Overview"
        current_path = current_heading
        current_lines: list[str] = []
        in_fence = False
        fence_marker = ""

        def finish_current() -> None:
            text = "\n".join(current_lines).strip()
            if text:
                raw_sections.append(
                    (current_level, current_heading, current_path, list(current_lines))
                )

        for line in body.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""
                current_lines.append(line)
                continue

            match = None if in_fence else cls.HEADING_PATTERN.match(line)
            if match is None:
                current_lines.append(line)
                continue

            finish_current()
            level = len(match.group(1))
            heading = cls._clean_heading(match.group(2)) or "Untitled section"
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, heading))
            current_level = level
            current_heading = heading
            current_path = " > ".join(item[1] for item in heading_stack)
            current_lines = []

        finish_current()

        occurrence: dict[str, int] = {}
        sections: list[_WikiSection] = []
        for level, heading, heading_path, lines in raw_sections:
            text = "\n".join(lines).strip()
            if not text or heading.casefold() in cls.PROVENANCE_HEADINGS:
                continue
            identity = heading_path.casefold()
            occurrence[identity] = occurrence.get(identity, 0) + 1
            section_id = hashlib.sha256(
                f"{identity}\0{occurrence[identity]}".encode("utf-8")
            ).hexdigest()[:16]
            sections.append(
                _WikiSection(
                    section_id=section_id,
                    heading=heading,
                    heading_path=heading_path,
                    level=level,
                    text=text,
                    sha256=cls._digest(text),
                )
            )

        if sections:
            return sections

        fallback_text = body.strip() or fallback_title.strip() or "Overview"
        heading = fallback_title.strip() or "Overview"
        return [
            _WikiSection(
                section_id=hashlib.sha256(b"overview\0\1").hexdigest()[:16],
                heading=heading,
                heading_path=heading,
                level=0,
                text=fallback_text,
                sha256=cls._digest(fallback_text),
            )
        ]

    @staticmethod
    def _section_document_text(path: str, title: str, section: _WikiSection) -> str:
        return (
            f"Wiki path: {path}\nTitle: {title}\n"
            f"Section: {section.heading_path}\n\n{section.text}"
        )

    @staticmethod
    def _excerpt(text: str, *, limit: int = 360) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1].rstrip() + "…"

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

    @staticmethod
    def _cached_sections(value: object) -> dict[str, Mapping[str, object]]:
        if not isinstance(value, Mapping):
            return {}
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list):
            return {}
        return {
            str(item.get("section_id")): item
            for item in raw_sections
            if isinstance(item, Mapping) and isinstance(item.get("section_id"), str)
        }

    def refresh(self) -> IndexRefreshResult:
        """Embed changed Wiki sections and remove stale page/section entries."""

        if self.embedder is None:
            pages_total = len(self.repository.list_wiki_pages())
            return IndexRefreshResult(pages_total, 0, pages_total, 0, 0)

        with self._refresh_lock, self._file_lock:
            cache = self._load_cache()
            cached_pages = dict(cache.get("pages", {}))
            pages = self.repository.list_wiki_pages()
            known_paths = {page.path for page in pages}
            removed_paths = set(cached_pages) - known_paths
            pages_removed = len(removed_paths)
            sections_removed = sum(
                len(self._cached_sections(cached_pages[path])) for path in removed_paths
            )
            changed = pages_removed > 0 or not self.cache_path.is_file()
            input_tokens = 0
            pages_embedded = 0
            sections_total = 0
            sections_embedded = 0
            sections_cached = 0
            retained: dict[str, object] = {}

            for page in pages:
                content = self.repository.read_wiki_page(page.path)
                page_digest = self._digest(content)
                sections = self._sections(content, fallback_title=page.title)
                sections_total += len(sections)
                cached_page = cached_pages.get(page.path)
                cached_by_id = self._cached_sections(cached_page)
                current_ids = {section.section_id for section in sections}
                sections_removed += len(set(cached_by_id) - current_ids)
                page_had_embedding = False
                retained_sections: list[dict[str, object]] = []

                for section in sections:
                    cached_section = cached_by_id.get(section.section_id)
                    vector = None
                    if (
                        isinstance(cached_section, Mapping)
                        and cached_section.get("sha256") == section.sha256
                    ):
                        vector = self._valid_vector(
                            cached_section.get("embedding"), self.embedder.dimensions
                        )
                    if vector is None:
                        result = self.embedder.embed(
                            self._section_document_text(page.path, page.title, section)
                        )
                        vector = result.vector
                        input_tokens += result.input_tokens
                        sections_embedded += 1
                        page_had_embedding = True
                        changed = True
                    else:
                        sections_cached += 1

                    retained_sections.append(
                        {
                            "section_id": section.section_id,
                            "sha256": section.sha256,
                            "heading": section.heading,
                            "heading_path": section.heading_path,
                            "level": section.level,
                            "excerpt": self._excerpt(section.text),
                            "embedding": list(vector),
                        }
                    )

                if page_had_embedding:
                    pages_embedded += 1
                if (
                    not isinstance(cached_page, Mapping)
                    or cached_page.get("sha256") != page_digest
                    or set(cached_by_id) != current_ids
                ):
                    changed = True
                retained[page.path] = {
                    "sha256": page_digest,
                    "title": page.title,
                    "summary": page.summary,
                    "sections": retained_sections,
                }

            if changed:
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
                pages_embedded=pages_embedded,
                pages_cached=len(pages) - pages_embedded,
                pages_removed=pages_removed,
                embedding_input_tokens=input_tokens,
                sections_total=sections_total,
                sections_embedded=sections_embedded,
                sections_cached=sections_cached,
                sections_removed=sections_removed,
            )

    def _semantic_results(
        self, query: str, *, limit: int
    ) -> tuple[list[WikiSearchResult], dict[str, tuple[SectionMatch, ...]], int]:
        if self.embedder is None:
            return [], {}, 0
        refresh = self.refresh()
        with self._file_lock:
            cache = self._load_cache()
        cached_pages = cache.get("pages", {})
        if not isinstance(cached_pages, Mapping) or not cached_pages:
            return [], {}, refresh.embedding_input_tokens

        query_result = self.embedder.embed(query)
        page_matches: dict[str, list[SectionMatch]] = {}
        page_titles: dict[str, str] = {}
        for path, value in cached_pages.items():
            if not isinstance(path, str) or not isinstance(value, Mapping):
                continue
            title = str(value.get("title", path))
            raw_sections = value.get("sections")
            if not isinstance(raw_sections, list):
                continue
            for section in raw_sections:
                if not isinstance(section, Mapping):
                    continue
                vector = self._valid_vector(
                    section.get("embedding"), self.embedder.dimensions
                )
                if vector is None:
                    continue
                similarity = sum(
                    left * right for left, right in zip(query_result.vector, vector)
                )
                page_titles[path] = title
                page_matches.setdefault(path, []).append(
                    SectionMatch(
                        section_id=str(section.get("section_id", "")),
                        heading=str(section.get("heading", "Overview")),
                        heading_path=str(section.get("heading_path", "Overview")),
                        excerpt=str(section.get("excerpt", "")),
                        semantic_score=round(similarity, 6),
                    )
                )

        matched_sections: dict[str, tuple[SectionMatch, ...]] = {}
        semantic_results: list[WikiSearchResult] = []
        for path, matches in page_matches.items():
            matches.sort(
                key=lambda item: (-item.semantic_score, item.heading_path.casefold())
            )
            top_matches = tuple(matches[: self.MAX_MATCHED_SECTIONS_PER_PAGE])
            matched_sections[path] = top_matches
            best = top_matches[0]
            semantic_results.append(
                WikiSearchResult(
                    path=path,
                    title=page_titles[path],
                    excerpt=f"[{best.heading_path}] {best.excerpt}".strip(),
                    score=best.semantic_score,
                )
            )

        semantic_results.sort(key=lambda item: (-item.score, item.path.casefold()))
        bounded_limit = max(1, min(int(limit), 20))
        selected = semantic_results[:bounded_limit]
        selected_paths = {item.path for item in selected}
        return (
            selected,
            {
                path: matches
                for path, matches in matched_sections.items()
                if path in selected_paths
            },
            refresh.embedding_input_tokens + query_result.input_tokens,
        )

    @staticmethod
    def _rank_map(results: Sequence[WikiSearchResult]) -> dict[str, int]:
        return {item.path: rank for rank, item in enumerate(results, start=1)}

    def _combine(
        self,
        lexical: list[WikiSearchResult],
        semantic: list[WikiSearchResult],
        matched_sections: Mapping[str, tuple[SectionMatch, ...]],
        *,
        limit: int,
    ) -> tuple[list[WikiSearchResult], tuple[PageSearchDiagnostic, ...]]:
        # Weighted reciprocal-rank fusion avoids comparing unrelated lexical
        # count and cosine-similarity scales.
        scores: dict[str, float] = {}
        items: dict[str, WikiSearchResult] = {}
        lexical_ranks = self._rank_map(lexical)
        semantic_ranks = self._rank_map(semantic)
        for rank, item in enumerate(lexical, start=1):
            scores[item.path] = scores.get(item.path, 0.0) + self.lexical_weight / (60 + rank)
            items[item.path] = item
        for rank, item in enumerate(semantic, start=1):
            scores[item.path] = scores.get(item.path, 0.0) + self.semantic_weight / (60 + rank)
            # A semantic excerpt identifies the best matching section and is
            # more useful for navigation than a page-level lexical fragment.
            items[item.path] = item
        ranked_paths = sorted(scores, key=lambda path: (-scores[path], path.casefold()))
        bounded_paths = ranked_paths[: max(1, min(int(limit), 20))]
        results = [
            WikiSearchResult(
                path=path,
                title=items[path].title,
                excerpt=items[path].excerpt,
                score=round(scores[path] * 10_000, 6),
            )
            for path in bounded_paths
        ]
        diagnostics = tuple(
            PageSearchDiagnostic(
                path=path,
                title=items[path].title,
                final_rank=rank,
                fused_score=round(scores[path] * 10_000, 6),
                lexical_rank=lexical_ranks.get(path),
                semantic_rank=semantic_ranks.get(path),
                matched_sections=matched_sections.get(path, ()),
            )
            for rank, path in enumerate(bounded_paths, start=1)
        )
        return results, diagnostics

    @staticmethod
    def _lexical_diagnostics(
        results: Sequence[WikiSearchResult],
    ) -> tuple[PageSearchDiagnostic, ...]:
        return tuple(
            PageSearchDiagnostic(
                path=item.path,
                title=item.title,
                final_rank=rank,
                fused_score=item.score,
                lexical_rank=rank,
            )
            for rank, item in enumerate(results, start=1)
        )

    def search(self, query: str, *, limit: int = 8) -> SearchResponse:
        bounded_limit = max(1, min(int(limit), 20))
        candidate_limit = min(20, max(bounded_limit * 3, bounded_limit))
        lexical = self.repository.search_wiki(query, limit=candidate_limit)
        if self.embedder is None:
            results = lexical[:bounded_limit]
            return SearchResponse(
                tuple(results), "lexical", diagnostics=self._lexical_diagnostics(results)
            )
        try:
            semantic, matched_sections, input_tokens = self._semantic_results(
                query, limit=candidate_limit
            )
            combined, diagnostics = self._combine(
                lexical, semantic, matched_sections, limit=bounded_limit
            )
            return SearchResponse(
                tuple(combined), "hybrid_section", input_tokens, diagnostics
            )
        except (EmbeddingError, RepositoryError, WikiSearchError, OSError, ValueError) as exc:
            # Searching must remain available during embedding outages. Log only
            # the safe exception type and use deterministic lexical results.
            logger.warning(
                "Semantic Wiki search unavailable; using lexical fallback (%s).",
                type(exc).__name__,
            )
            results = lexical[:bounded_limit]
            return SearchResponse(
                tuple(results),
                "lexical_fallback",
                diagnostics=self._lexical_diagnostics(results),
            )
