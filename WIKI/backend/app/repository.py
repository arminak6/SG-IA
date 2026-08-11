"""Filesystem boundary for immutable raw sources and the generated wiki."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import yaml
from filelock import FileLock

from .extraction import DoclingDocumentExtractor, DocumentExtractionError, DocumentExtractor


class RepositoryError(RuntimeError):
    """Base class for repository failures."""


class UnsafePathError(RepositoryError):
    """Raised when a requested path escapes its assigned directory."""


class SourceReadError(RepositoryError):
    """Raised when a raw source cannot safely be read as text."""


@dataclass(frozen=True)
class RawDocument:
    relative_path: str
    source_path: str
    size_bytes: int
    modified_at: str
    is_ingested: bool

    @property
    def status(self) -> str:
        return "Ingested" if self.is_ingested else "Pending"

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "source_path": self.source_path,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "is_ingested": self.is_ingested,
            "status": self.status,
        }


@dataclass(frozen=True)
class WikiPage:
    path: str
    title: str
    summary: str
    source_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "source_paths": list(self.source_paths),
        }


@dataclass(frozen=True)
class WikiSearchResult:
    path: str
    title: str
    excerpt: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "title": self.title,
            "excerpt": self.excerpt,
            "score": self.score,
        }


class WikiRepository:
    """Restrict all filesystem access to ``backend/raw`` and ``backend/wiki``.

    The only raw write operation creates a new immutable Markdown file in the
    dedicated manager-actions directory after application confirmation.
    Wiki writes accept Markdown only, validate containment after symlink
    resolution, and use ``os.replace`` so readers see either the old complete
    file or the new complete file.
    """

    SYSTEM_PAGES = frozenset({"index.md", "log.md"})
    MANIFEST_FILENAME = ".ingestion-manifest.json"
    LOCK_FILENAME = ".ingestion.lock"
    PAGE_TYPES = frozenset({"source", "concept", "entity", "synthesis"})
    DIRECTORY_PAGE_TYPES = {
        "sources": "source",
        "concepts": "concept",
        "entities": "entity",
        "syntheses": "synthesis",
    }
    MARKDOWN_LINK_PATTERN = re.compile(
        r"(?<!!)\[[^\]]+\]\(\s*(?:<([^>]+)>|([^\s)]+))", re.MULTILINE
    )
    TEXT_SOURCE_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".csv"})
    BINARY_SOURCE_SUFFIXES = DoclingDocumentExtractor.SUPPORTED_SUFFIXES
    SOURCE_SUFFIXES = TEXT_SOURCE_SUFFIXES | BINARY_SOURCE_SUFFIXES
    MANAGER_ACTIONS_DIR = "manager-actions"
    MANAGER_CORRECTIONS_DIR = MANAGER_ACTIONS_DIR

    def __init__(
        self,
        backend_root: Path | str,
        *,
        max_source_bytes: int = 25_000_000,
        max_extracted_characters: int = 600_000,
        document_extractor: DocumentExtractor | None = None,
    ) -> None:
        self.backend_root = Path(backend_root).resolve()
        self.raw_root = (self.backend_root / "raw").resolve()
        self.wiki_root = (self.backend_root / "wiki").resolve()
        self.schema_path = self.backend_root / "AGENTS.md"
        self.max_source_bytes = max_source_bytes
        self.max_extracted_characters = max_extracted_characters
        self.manifest_path = self.wiki_root / self.MANIFEST_FILENAME
        self._ingestion_file_lock = FileLock(str(self.wiki_root / self.LOCK_FILENAME))
        self.document_extractor = document_extractor or DoclingDocumentExtractor(
            max_extracted_characters=max_extracted_characters
        )
        self._write_lock = threading.RLock()

    @contextmanager
    def ingestion_lock(self):
        """Serialize ingestion across API workers that share this wiki directory."""

        self.wiki_root.mkdir(parents=True, exist_ok=True)
        with self._ingestion_file_lock:
            yield

    @staticmethod
    def _normalize_relative(value: str, *, prefix: str | None = None) -> str:
        if not isinstance(value, str):
            raise UnsafePathError("Path must be text.")
        normalized = value.strip().replace("\\", "/")
        if prefix:
            marker = f"{prefix}/"
            if not normalized.startswith(marker):
                raise UnsafePathError(f"Path must start with {marker}.")
            normalized = normalized[len(marker) :]
        if not normalized or any(character in normalized for character in ("\x00", "\r", "\n")):
            raise UnsafePathError("Path is empty or invalid.")
        windows_path = PureWindowsPath(normalized)
        if normalized.startswith(("/", "//")) or windows_path.is_absolute() or windows_path.drive:
            raise UnsafePathError("Absolute paths are not allowed.")
        raw_parts = normalized.split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise UnsafePathError("Path traversal and empty path segments are not allowed.")
        path = PurePosixPath(normalized)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise UnsafePathError("Path must remain inside its assigned directory.")
        return path.as_posix()

    @classmethod
    def normalize_source_path(cls, source_path: str) -> str:
        relative = cls._normalize_relative(source_path, prefix="raw")
        return f"raw/{relative}"

    @classmethod
    def normalize_wiki_path(cls, wiki_path: str, *, allow_system: bool = True) -> str:
        relative = cls._normalize_relative(wiki_path)
        if any(part.startswith(".") for part in PurePosixPath(relative).parts):
            raise UnsafePathError("Hidden wiki paths are not allowed.")
        if PurePosixPath(relative).suffix.casefold() != ".md":
            raise UnsafePathError("Wiki pages must use the .md extension.")
        if not allow_system and relative.casefold() in cls.SYSTEM_PAGES:
            raise UnsafePathError("System wiki pages are maintained by the application.")
        return relative

    @staticmethod
    def _contained_path(root: Path, relative: str) -> Path:
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise UnsafePathError("Resolved path escapes its assigned directory.") from exc
        return candidate

    def _raw_file(self, source_path: str) -> tuple[str, Path]:
        normalized = self.normalize_source_path(source_path)
        relative = normalized.removeprefix("raw/")
        target = self._contained_path(self.raw_root, relative)
        if not target.is_file():
            raise SourceReadError(f"Raw source does not exist: {normalized}")
        return normalized, target

    def _wiki_file(self, wiki_path: str, *, allow_system: bool = True) -> tuple[str, Path]:
        normalized = self.normalize_wiki_path(wiki_path, allow_system=allow_system)
        return normalized, self._contained_path(self.wiki_root, normalized)

    @staticmethod
    def _is_hidden(relative: Path) -> bool:
        return any(part.startswith(".") for part in relative.parts)

    @staticmethod
    def _has_exact_reference(text: str, source_path: str) -> bool:
        # Reject a filename-prefix match such as raw/a.md within raw/a.md.bak.
        boundary_chars = r"\w./-"
        pattern = rf"(?<![{boundary_chars}]){re.escape(source_path)}(?![{boundary_chars}])"
        return re.search(pattern, text.replace("\\", "/"), flags=re.IGNORECASE) is not None

    def read_schema(self) -> str:
        try:
            return self.schema_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise RepositoryError("Could not read backend/AGENTS.md.") from exc

    def read_raw(self, source_path: str) -> str:
        normalized, target = self._raw_file(source_path)
        suffix = target.suffix.casefold()
        if suffix not in self.SOURCE_SUFFIXES:
            raise SourceReadError(
                f"Unsupported source type for {normalized}; supported types are Markdown, TXT, JSON, "
                "CSV, PDF, DOCX, and PPTX."
            )
        try:
            size = target.stat().st_size
            if size > self.max_source_bytes:
                raise SourceReadError(
                    f"Raw source is too large to ingest ({size} bytes; limit {self.max_source_bytes})."
                )
            if suffix in self.TEXT_SOURCE_SUFFIXES:
                content = target.read_text(encoding="utf-8-sig")
                if len(content) > self.max_extracted_characters:
                    raise SourceReadError(
                        f"Extracted content for {normalized} exceeds the safety limit "
                        f"({len(content)} characters; limit {self.max_extracted_characters})."
                    )
                return content
            return self.document_extractor.extract(target, source_path=normalized)
        except UnicodeDecodeError as exc:
            raise SourceReadError(f"Raw source is not valid UTF-8 text: {normalized}") from exc
        except DocumentExtractionError as exc:
            raise SourceReadError(str(exc)) from exc
        except OSError as exc:
            raise SourceReadError(f"Could not read raw source: {normalized}") from exc

    def raw_exists(self, source_path: str) -> bool:
        try:
            self._raw_file(source_path)
        except RepositoryError:
            return False
        return True

    def create_manager_action_source(self, filename: str, content: str) -> str:
        """Create one immutable, application-authored manager knowledge source."""

        relative_name = self._normalize_relative(filename)
        if len(PurePosixPath(relative_name).parts) != 1:
            raise UnsafePathError("Manager action filename cannot contain directories.")
        if PurePosixPath(relative_name).suffix.casefold() != ".md":
            raise UnsafePathError("Manager action sources must be Markdown files.")
        if not isinstance(content, str) or not content.strip():
            raise RepositoryError("Manager action source content cannot be empty.")
        if len(content) > self.max_extracted_characters:
            raise RepositoryError("Manager action source exceeds the text safety limit.")

        action_root = self._contained_path(self.raw_root, self.MANAGER_ACTIONS_DIR)
        target = self._contained_path(action_root, relative_name)
        try:
            action_root.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content.rstrip() + "\n")
        except FileExistsError as exc:
            raise RepositoryError("Manager action source already exists.") from exc
        except OSError as exc:
            raise RepositoryError(
                "Could not create the manager action source. Check the writable "
                "manager-actions mount."
            ) from exc
        return f"raw/{self.MANAGER_ACTIONS_DIR}/{relative_name}"

    def create_manager_correction_source(self, filename: str, content: str) -> str:
        """Backward-compatible alias for the manager-action source writer."""

        return self.create_manager_action_source(filename, content)

    def source_digest(self, source_path: str) -> str:
        """Return a content hash so an edited source becomes pending again."""

        _, target = self._raw_file(source_path)
        digest = hashlib.sha256()
        try:
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise SourceReadError(f"Could not hash raw source: {source_path}") from exc
        return digest.hexdigest()

    def _read_manifest(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError("Could not read the ingestion manifest.") from exc
        sources = payload.get("sources") if isinstance(payload, Mapping) else None
        if not isinstance(sources, Mapping):
            raise RepositoryError("The ingestion manifest has an invalid format.")
        return {
            str(path): dict(record)
            for path, record in sources.items()
            if isinstance(path, str) and isinstance(record, Mapping)
        }

    def _manifest_content(self, sources: Mapping[str, Mapping[str, object]]) -> str:
        payload = {
            "version": 1,
            "sources": {
                path: dict(record)
                for path, record in sorted(sources.items(), key=lambda item: item[0].casefold())
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    def _content_page_paths(self) -> list[Path]:
        if not self.wiki_root.is_dir():
            return []
        pages: list[Path] = []
        for path in self.wiki_root.rglob("*.md"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.wiki_root)
            if self._is_hidden(relative) or relative.as_posix().casefold() in self.SYSTEM_PAGES:
                continue
            try:
                self._contained_path(self.wiki_root, relative.as_posix())
            except UnsafePathError:
                continue
            pages.append(path)
        return sorted(pages, key=lambda item: item.relative_to(self.wiki_root).as_posix().casefold())

    def _content_page_contents(self) -> dict[str, str]:
        """Read each knowledge page once for repository status operations."""

        contents: dict[str, str] = {}
        with self._write_lock:
            for path in self._content_page_paths():
                relative = path.relative_to(self.wiki_root).as_posix()
                try:
                    contents[relative] = path.read_text(encoding="utf-8-sig")
                except OSError:
                    continue
        return contents

    def provenance_pages(self, source_path: str, *, overlays: Mapping[str, str] | None = None) -> list[str]:
        source = self.normalize_source_path(source_path)
        contents = self._content_page_contents()
        if overlays:
            for page_path, content in overlays.items():
                normalized = self.normalize_wiki_path(page_path, allow_system=False)
                contents[normalized] = content
        return sorted(
            (path for path, content in contents.items() if self._has_exact_reference(content, source)),
            key=str.casefold,
        )

    def is_ingested(self, source_path: str) -> bool:
        source = self.normalize_source_path(source_path)
        with self._write_lock:
            record = self._read_manifest().get(source)
            if not record or record.get("sha256") != self.source_digest(source):
                return False
            pages = record.get("pages")
            if not isinstance(pages, list) or not pages:
                return False
            for page in pages:
                try:
                    normalized, target = self._wiki_file(str(page), allow_system=False)
                    content = target.read_text(encoding="utf-8-sig")
                except (OSError, RepositoryError):
                    return False
                if normalized != page or not self._has_exact_reference(content, source):
                    return False
            return True

    def list_raw_documents(self) -> list[RawDocument]:
        return self._list_raw_documents({})

    def _list_raw_documents(self, page_contents: Mapping[str, str]) -> list[RawDocument]:
        if not self.raw_root.is_dir():
            return []
        documents: list[RawDocument] = []
        for path in self.raw_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.raw_root)
            if self._is_hidden(relative):
                continue
            if path.suffix.casefold() not in self.SOURCE_SUFFIXES:
                continue
            relative_path = relative.as_posix()
            try:
                resolved = self._contained_path(self.raw_root, relative_path)
                stat = resolved.stat()
            except (OSError, UnsafePathError):
                continue
            source_path = f"raw/{relative_path}"
            documents.append(
                RawDocument(
                    relative_path=relative_path,
                    source_path=source_path,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    is_ingested=self.is_ingested(source_path),
                )
            )
        return sorted(documents, key=lambda item: item.relative_path.casefold())

    def count_wiki_pages(self) -> int:
        """Return the number of generated knowledge pages without parsing them."""

        return len(self._content_page_paths())

    def read_wiki_page(self, wiki_path: str, *, overlays: Mapping[str, str] | None = None) -> str:
        normalized, target = self._wiki_file(wiki_path)
        if overlays and normalized in overlays:
            return overlays[normalized]
        with self._write_lock:
            try:
                return target.read_text(encoding="utf-8-sig")
            except FileNotFoundError as exc:
                raise RepositoryError(f"Wiki page does not exist: {normalized}") from exc
            except OSError as exc:
                raise RepositoryError(f"Could not read wiki page: {normalized}") from exc

    @staticmethod
    def _title(content: str, fallback: str) -> str:
        match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
        return match.group(1).strip() if match else PurePosixPath(fallback).stem.replace("-", " ").title()

    @staticmethod
    def _summary(content: str, limit: int = 180) -> str:
        lines = content.splitlines()
        in_frontmatter = bool(lines and lines[0].strip() == "---")
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if index == 0 and in_frontmatter:
                continue
            if line == "---" and in_frontmatter:
                in_frontmatter = False
                continue
            if in_frontmatter or not line or line.startswith(("#", "- raw/", "* raw/")):
                continue
            cleaned = re.sub(r"\s+", " ", line)
            return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"
        return "No summary available."

    def page_source_paths(self, wiki_path: str, *, content: str | None = None) -> list[str]:
        page_text = self.read_wiki_page(wiki_path) if content is None else content
        return self._source_paths_for_content(page_text, self.list_raw_documents())

    def _source_paths_for_content(
        self, content: str, documents: Sequence[RawDocument]
    ) -> list[str]:
        return [
            document.source_path
            for document in documents
            if self._has_exact_reference(content, document.source_path)
        ]

    def list_wiki_pages(self, *, include_system: bool = False) -> list[WikiPage]:
        if not self.wiki_root.is_dir():
            return []
        content_pages = self._content_page_contents()
        documents = self._list_raw_documents(content_pages)
        page_contents: dict[str, str] = dict(content_pages)
        if include_system:
            for system_page in sorted(self.SYSTEM_PAGES):
                path = self.wiki_root / system_page
                if path.is_file():
                    try:
                        page_contents[system_page] = path.read_text(encoding="utf-8-sig")
                    except OSError:
                        continue
        pages: list[WikiPage] = []
        for relative, content in sorted(page_contents.items(), key=lambda item: item[0].casefold()):
            pages.append(
                WikiPage(
                    path=relative,
                    title=self._title(content, relative),
                    summary=self._summary(content),
                    source_paths=tuple(self._source_paths_for_content(content, documents)),
                )
            )
        return pages

    def search_wiki(
        self,
        query: str,
        *,
        limit: int = 8,
        overlays: Mapping[str, str] | None = None,
    ) -> list[WikiSearchResult]:
        terms = sorted(set(re.findall(r"[\w-]{2,}", query.casefold())))
        if not terms:
            return []
        documents: dict[str, str] = {}
        for path in self._content_page_paths():
            relative = path.relative_to(self.wiki_root).as_posix()
            try:
                documents[relative] = path.read_text(encoding="utf-8-sig")
            except OSError:
                continue
        if overlays:
            for page_path, content in overlays.items():
                documents[self.normalize_wiki_path(page_path, allow_system=False)] = content

        matches: list[WikiSearchResult] = []
        query_folded = query.casefold().strip()
        for path, content in documents.items():
            haystack = f"{path}\n{content}".casefold()
            score = sum(haystack.count(term) for term in terms)
            if query_folded and query_folded in haystack:
                score += 10
            if score <= 0:
                continue
            excerpt = self._matching_excerpt(content, terms)
            matches.append(
                WikiSearchResult(
                    path=path,
                    title=self._title(content, path),
                    excerpt=excerpt,
                    score=score,
                )
            )
        matches.sort(key=lambda item: (-item.score, item.path.casefold()))
        return matches[: max(1, min(int(limit), 20))]

    @staticmethod
    def _matching_excerpt(content: str, terms: Sequence[str], limit: int = 320) -> str:
        compact = re.sub(r"\s+", " ", content).strip()
        folded = compact.casefold()
        positions = [folded.find(term) for term in terms if folded.find(term) >= 0]
        start = max(0, (min(positions) if positions else 0) - 80)
        excerpt = compact[start : start + limit]
        if start:
            excerpt = "…" + excerpt
        if start + limit < len(compact):
            excerpt += "…"
        return excerpt

    def _validate_markdown(self, content: str, *, page_path: str | None = None) -> str:
        if not isinstance(content, str) or not content.strip():
            raise RepositoryError("Wiki page content cannot be empty.")
        if "\x00" in content:
            raise RepositoryError("Wiki page contains an invalid null character.")
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
        if not normalized.startswith("---\n"):
            raise RepositoryError("Wiki knowledge pages must begin with YAML frontmatter.")
        frontmatter_end = normalized.find("\n---\n", 4)
        if frontmatter_end < 0:
            raise RepositoryError("Wiki page YAML frontmatter is not closed.")
        frontmatter = normalized[4:frontmatter_end]
        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as exc:
            raise RepositoryError("Wiki page frontmatter is not valid YAML.") from exc
        if not isinstance(metadata, Mapping):
            raise RepositoryError("Wiki page frontmatter must be a YAML object.")
        for required_field in ("title", "page_type", "updated", "sources"):
            if required_field not in metadata:
                raise RepositoryError(
                    f"Wiki page frontmatter is missing required field: {required_field}."
                )
        if not isinstance(metadata["title"], str) or not metadata["title"].strip():
            raise RepositoryError("Wiki page title must be non-empty text.")
        page_type = metadata["page_type"]
        if page_type not in self.PAGE_TYPES:
            allowed = ", ".join(sorted(self.PAGE_TYPES))
            raise RepositoryError(f"Wiki page_type must be one of: {allowed}.")
        updated = metadata["updated"]
        if isinstance(updated, datetime):
            valid_date = False
        elif isinstance(updated, date):
            valid_date = True
        elif isinstance(updated, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
            try:
                date.fromisoformat(updated)
                valid_date = True
            except ValueError:
                valid_date = False
        else:
            valid_date = False
        if not valid_date:
            raise RepositoryError("Wiki page updated must be a valid YYYY-MM-DD date.")
        source_values = metadata["sources"]
        if not isinstance(source_values, list) or not source_values:
            raise RepositoryError("Wiki page sources must be a non-empty YAML list.")
        sources: list[str] = []
        for value in source_values:
            if not isinstance(value, str):
                raise RepositoryError("Every wiki source must be a raw/... text path.")
            source = self.normalize_source_path(value)
            if source != value.replace("\\", "/"):
                raise RepositoryError(f"Wiki source path is not normalized: {value}")
            if not self.raw_exists(source):
                raise RepositoryError(f"Wiki source does not exist: {source}")
            sources.append(source)
        if len(set(sources)) != len(sources):
            raise RepositoryError("Wiki page sources must not contain duplicates.")
        if page_path:
            normalized_path = self.normalize_wiki_path(page_path, allow_system=False)
            folder = PurePosixPath(normalized_path).parts[0]
            expected_type = self.DIRECTORY_PAGE_TYPES.get(folder)
            if expected_type is None:
                raise RepositoryError(
                    "Wiki knowledge pages must be stored under sources, concepts, entities, "
                    "or syntheses."
                )
            if page_type != expected_type:
                raise RepositoryError(
                    f"Wiki page {normalized_path} must use page_type: {expected_type}."
                )
        if not re.search(r"^#\s+\S", normalized, flags=re.MULTILINE):
            raise RepositoryError("Wiki pages must contain a level-one Markdown heading.")
        sources_heading = re.search(r"^## Sources\s*$", normalized, flags=re.MULTILINE)
        if not sources_heading:
            raise RepositoryError("Wiki pages must contain an exact '## Sources' section.")
        section_start = sources_heading.end()
        next_heading = re.search(r"^##\s+", normalized[section_start:], flags=re.MULTILINE)
        section_end = section_start + next_heading.start() if next_heading else len(normalized)
        sources_section = normalized[section_start:section_end]
        missing_sources = [
            source for source in sources if not self._has_exact_reference(sources_section, source)
        ]
        if missing_sources:
            raise RepositoryError(
                "Wiki Sources section is missing frontmatter provenance: "
                + ", ".join(missing_sources)
            )
        return normalized

    def _link_issues(
        self, page_path: str, content: str, *, staged_paths: Iterable[str] = ()
    ) -> list[str]:
        staged = {self.normalize_wiki_path(path, allow_system=False) for path in staged_paths}
        _, page_file = self._wiki_file(page_path)
        issues: list[str] = []
        for match in self.MARKDOWN_LINK_PATTERN.finditer(content):
            target_text = (match.group(1) or match.group(2) or "").strip()
            parsed = urlsplit(target_text)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            link_path = unquote(parsed.path).replace("\\", "/")
            candidate = (page_file.parent / Path(*PurePosixPath(link_path).parts)).resolve(
                strict=False
            )
            try:
                relative_wiki = candidate.relative_to(self.wiki_root).as_posix()
            except ValueError:
                relative_wiki = None
            if relative_wiki is not None:
                if relative_wiki in staged or candidate.is_file():
                    continue
                issues.append(f"Broken wiki link '{target_text}'.")
                continue
            try:
                candidate.relative_to(self.raw_root)
            except ValueError:
                issues.append(f"Local link escapes the wiki/raw roots: '{target_text}'.")
                continue
            if not candidate.is_file():
                issues.append(f"Broken raw-source link '{target_text}'.")
        return issues

    def _validate_links(self, pages: Mapping[str, str]) -> None:
        for page_path, content in pages.items():
            issues = self._link_issues(page_path, content, staged_paths=pages)
            if issues:
                raise RepositoryError(f"{page_path}: {issues[0]}")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _prepare_wiki_pages(self, pages: Mapping[str, str]) -> dict[str, tuple[Path, str]]:
        prepared: dict[str, tuple[Path, str]] = {}
        for requested_path, content in pages.items():
            normalized, target = self._wiki_file(requested_path, allow_system=False)
            prepared[normalized] = (
                target,
                self._validate_markdown(content, page_path=normalized),
            )
        self._validate_links({path: value[1] for path, value in prepared.items()})
        return prepared

    @staticmethod
    def _snapshot(paths: Iterable[Path]) -> dict[Path, bytes | None]:
        snapshots: dict[Path, bytes | None] = {}
        for path in paths:
            try:
                snapshots[path] = path.read_bytes()
            except FileNotFoundError:
                snapshots[path] = None
        return snapshots

    def _restore_snapshot(self, snapshots: Mapping[Path, bytes | None]) -> None:
        errors: list[str] = []
        for path, content in snapshots.items():
            try:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    self._atomic_write_bytes(path, content)
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
        if errors:
            raise RepositoryError("Could not fully restore wiki transaction: " + "; ".join(errors))

    def _write_prepared_pages(
        self,
        prepared: Mapping[str, tuple[Path, str]],
        *,
        manifest_source: str | None = None,
    ) -> list[str]:
        if not prepared:
            return []
        paths_to_snapshot = [target for target, _ in prepared.values()]
        paths_to_snapshot.append(self.wiki_root / "index.md")
        if manifest_source:
            paths_to_snapshot.append(self.manifest_path)
        snapshots = self._snapshot(paths_to_snapshot)
        try:
            for normalized in sorted(prepared, key=str.casefold):
                target, content = prepared[normalized]
                self._atomic_write(target, content)
            self.rebuild_index()
            if manifest_source:
                source = self.normalize_source_path(manifest_source)
                manifest = self._read_manifest()
                provenance = self.provenance_pages(source)
                if not provenance:
                    raise RepositoryError(
                        f"Committed pages do not retain provenance for {source}."
                    )
                manifest[source] = {
                    "sha256": self.source_digest(source),
                    "pages": provenance,
                    "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                self._atomic_write(self.manifest_path, self._manifest_content(manifest))
        except Exception as exc:
            try:
                self._restore_snapshot(snapshots)
            except RepositoryError as restore_exc:
                raise RepositoryError(
                    f"Wiki transaction failed and rollback was incomplete: {restore_exc}"
                ) from exc
            if isinstance(exc, RepositoryError):
                raise
            raise RepositoryError("Wiki transaction failed; all changes were rolled back.") from exc
        return sorted(prepared, key=str.casefold)

    def write_wiki_pages(self, pages: Mapping[str, str]) -> list[str]:
        """Validate and atomically replace each staged content page.

        System pages cannot be supplied here.  ``index.md`` is regenerated
        deterministically after all staged pages have been committed.
        """

        prepared = self._prepare_wiki_pages(pages)
        if not prepared:
            return []

        with self._write_lock:
            return self._write_prepared_pages(prepared)

    def commit_ingestion(self, source_path: str, pages: Mapping[str, str]) -> list[str]:
        """Commit staged pages, rebuilt index, and source hash as one transaction."""

        source = self.normalize_source_path(source_path)
        prepared = self._prepare_wiki_pages(pages)
        if not prepared:
            return []
        with self._write_lock:
            return self._write_prepared_pages(prepared, manifest_source=source)

    def rebuild_index(self) -> None:
        with self._write_lock:
            pages = self.list_wiki_pages()
            lines = [
                "# Wiki Index",
                "",
                "This catalog is maintained automatically after every successful ingestion.",
                "",
                "## Pages",
                "",
            ]
            if not pages:
                lines.append("_No knowledge pages have been ingested yet._")
            else:
                for page in pages:
                    source_label = f"; {len(page.source_paths)} source(s)" if page.source_paths else ""
                    lines.append(f"- [{page.title}](<{page.path}>) — {page.summary}{source_label}")
            lines.append("")
            self._atomic_write(self.wiki_root / "index.md", "\n".join(lines))

    def _linked_wiki_pages(
        self, page_path: str, content: str, known_pages: Iterable[str]
    ) -> set[str]:
        known = set(known_pages)
        _, page_file = self._wiki_file(page_path)
        targets: set[str] = set()
        for match in self.MARKDOWN_LINK_PATTERN.finditer(content):
            target_text = (match.group(1) or match.group(2) or "").strip()
            parsed = urlsplit(target_text)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            candidate = (
                page_file.parent
                / Path(*PurePosixPath(unquote(parsed.path).replace("\\", "/")).parts)
            ).resolve(strict=False)
            try:
                relative = candidate.relative_to(self.wiki_root).as_posix()
            except ValueError:
                continue
            if relative in known and relative != page_path:
                targets.add(relative)
        return targets

    def wiki_graph(self) -> dict[str, object]:
        """Return the knowledge-page graph without counting index.md as an edge."""

        contents = self._content_page_contents()
        outgoing = {
            path: self._linked_wiki_pages(path, content, contents)
            for path, content in contents.items()
        }
        incoming: dict[str, set[str]] = {path: set() for path in contents}
        for source, targets in outgoing.items():
            for target in targets:
                incoming[target].add(source)
        nodes = [
            {
                "path": path,
                "title": self._title(contents[path], path),
                "incoming": sorted(incoming[path], key=str.casefold),
                "outgoing": sorted(outgoing[path], key=str.casefold),
                "incoming_count": len(incoming[path]),
                "outgoing_count": len(outgoing[path]),
                "isolated": not incoming[path] and not outgoing[path],
            }
            for path in sorted(contents, key=str.casefold)
        ]
        return {
            "summary": {
                "pages": len(nodes),
                "links": sum(len(targets) for targets in outgoing.values()),
                "without_incoming": sum(not incoming[path] for path in contents),
                "without_outgoing": sum(not outgoing[path] for path in contents),
                "isolated": sum(
                    not incoming[path] and not outgoing[path] for path in contents
                ),
            },
            "nodes": nodes,
        }

    @staticmethod
    def _related_link_line(source_path: str, target_path: str, target_title: str) -> str:
        start = PurePosixPath(source_path).parent.as_posix()
        relative = posixpath.relpath(target_path, start=start)
        safe_title = target_title.replace("[", "").replace("]", "").strip()
        return f"- [{safe_title}](<{relative}>)"

    def _add_related_link(
        self,
        source_path: str,
        content: str,
        target_path: str,
        target_title: str,
        known_pages: Iterable[str],
    ) -> tuple[str, bool]:
        if target_path in self._linked_wiki_pages(source_path, content, known_pages):
            return content, False
        link_line = self._related_link_line(source_path, target_path, target_title)
        related_heading = re.search(
            r"^##\s+Related(?:\s+(?:pages?|wiki pages|concepts?|entit(?:y|ies)))?\s*$",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if related_heading:
            section_start = related_heading.end()
            next_heading = re.search(r"^##\s+", content[section_start:], flags=re.MULTILINE)
            insertion = section_start + next_heading.start() if next_heading else len(content)
            prefix = content[:insertion].rstrip()
            suffix = content[insertion:].lstrip("\n")
            updated = f"{prefix}\n{link_line}\n\n{suffix}"
        else:
            sources_heading = re.search(r"^## Sources\s*$", content, flags=re.MULTILINE)
            if not sources_heading:
                raise RepositoryError(f"Wiki page has no Sources section: {source_path}")
            prefix = content[: sources_heading.start()].rstrip()
            suffix = content[sources_heading.start() :].lstrip("\n")
            updated = f"{prefix}\n\n## Related pages\n\n{link_line}\n\n{suffix}"
        updated = re.sub(
            r"^updated:\s*.*$",
            f"updated: {date.today().isoformat()}",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        return updated.rstrip() + "\n", True

    def apply_cross_links(
        self, pairs: Iterable[tuple[str, str]], *, bidirectional: bool = True
    ) -> dict[str, object]:
        """Add validated related-page links without allowing model-authored prose edits."""

        contents = self._content_page_contents()
        titles = {path: self._title(content, path) for path, content in contents.items()}
        normalized_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for source_value, target_value in pairs:
            source = self.normalize_wiki_path(source_value, allow_system=False)
            target = self.normalize_wiki_path(target_value, allow_system=False)
            if source == target:
                raise RepositoryError("A wiki page cannot link to itself.")
            if source not in contents or target not in contents:
                raise RepositoryError(f"Cross-link page does not exist: {source} -> {target}")
            key = tuple(sorted((source, target), key=str.casefold))
            if key not in seen:
                normalized_pairs.append((source, target))
                seen.add(key)

        updated_pages: dict[str, str] = {}
        applied_pairs: list[dict[str, str]] = []
        for source, target in normalized_pairs:
            changed = False
            current_source = updated_pages.get(source, contents[source])
            current_source, source_changed = self._add_related_link(
                source, current_source, target, titles[target], contents
            )
            if source_changed:
                updated_pages[source] = current_source
                changed = True
            if bidirectional:
                current_target = updated_pages.get(target, contents[target])
                current_target, target_changed = self._add_related_link(
                    target, current_target, source, titles[source], contents
                )
                if target_changed:
                    updated_pages[target] = current_target
                    changed = True
            if changed:
                applied_pairs.append({"source_path": source, "target_path": target})
        pages_written = self.write_wiki_pages(updated_pages) if updated_pages else []
        return {"pairs_added": applied_pairs, "pages_updated": pages_written}

    def apply_answer_fix_guidance(
        self,
        *,
        action_id: str,
        target_page: str,
        subject: str,
        guidance: str,
        evidence_pages: Sequence[str],
    ) -> list[str]:
        """Integrate verified answer guidance into an existing graph page.

        This changes only the derived Wiki representation. It preserves the
        complete page, adds all evidence provenance, and links the target page
        to every supporting page. No raw source is created.
        """

        target = self.normalize_wiki_path(target_page, allow_system=False)
        evidence = [
            self.normalize_wiki_path(path, allow_system=False)
            for path in evidence_pages
        ]
        if target not in evidence:
            raise RepositoryError("Answer-fix target must be one of its evidence pages.")
        content = self.read_wiki_page(target)
        marker = f"<!-- manager-answer-fix:{action_id} -->"
        if marker in content:
            return []

        all_sources = set(self.page_source_paths(target, content=content))
        for page in evidence:
            all_sources.update(self.page_source_paths(page))
        frontmatter_end = content.find("\n---\n", 4)
        if not content.startswith("---\n") or frontmatter_end < 0:
            raise RepositoryError("Wiki page frontmatter is invalid.")
        metadata = yaml.safe_load(content[4:frontmatter_end])
        if not isinstance(metadata, dict):
            raise RepositoryError("Wiki page frontmatter must be a YAML object.")
        body = content[frontmatter_end + 5 :]
        metadata["updated"] = date.today().isoformat()
        metadata["sources"] = sorted(all_sources, key=str.casefold)
        rendered_frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        body = body.strip()

        related_lines: list[str] = []
        for page in sorted(set(evidence), key=str.casefold):
            if page == target:
                continue
            title = self._title(self.read_wiki_page(page), page)
            related_lines.append(self._related_link_line(target, page, title))
        related_text = "\n".join(related_lines)
        safe_subject = re.sub(r"\s+", " ", subject).strip()
        safe_guidance = guidance.strip()
        section = (
            f"## Manager-reviewed guidance: {safe_subject}\n\n"
            f"{marker}\n\n{safe_guidance}"
        )
        if related_text:
            section += f"\n\nSupporting Wiki pages:\n\n{related_text}"

        sources_heading = re.search(r"^## Sources\s*$", body, flags=re.MULTILINE)
        if not sources_heading:
            raise RepositoryError(f"Wiki page has no Sources section: {target}")
        before_sources = body[: sources_heading.start()].rstrip()
        sources_section = "## Sources\n\n" + "\n".join(
            f"- {source}" for source in sorted(all_sources, key=str.casefold)
        )
        updated = (
            f"---\n{rendered_frontmatter}\n---\n\n"
            f"{before_sources}\n\n{section}\n\n{sources_section}\n"
        )
        written = self.write_wiki_pages({target: updated})
        link_pairs = [(target, page) for page in evidence if page != target]
        if link_pairs:
            linked = self.apply_cross_links(link_pairs, bidirectional=True)
            for page in linked.get("pages_updated", []):
                if isinstance(page, str) and page not in written:
                    written.append(page)
        return sorted(written, key=str.casefold)

    def lint_wiki(self) -> dict[str, object]:
        """Run deterministic schema, provenance, link, and index checks."""

        issues: list[dict[str, str]] = []
        contents = self._content_page_contents()
        for page_path, content in contents.items():
            try:
                self._validate_markdown(content, page_path=page_path)
            except RepositoryError as exc:
                issues.append(
                    {
                        "severity": "error",
                        "code": "invalid_page",
                        "path": page_path,
                        "message": str(exc),
                    }
                )
            for message in self._link_issues(page_path, content):
                issues.append(
                    {
                        "severity": "error",
                        "code": "broken_link",
                        "path": page_path,
                        "message": message,
                    }
                )

        index_path = self.wiki_root / "index.md"
        try:
            index_content = index_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            index_content = ""
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_index",
                    "path": "index.md",
                    "message": "The generated wiki index is missing.",
                }
            )
        except OSError as exc:
            raise RepositoryError("Could not read wiki index.") from exc

        indexed_pages: set[str] = set()
        for match in self.MARKDOWN_LINK_PATTERN.finditer(index_content):
            target = unquote((match.group(1) or match.group(2) or "").strip())
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            normalized = posixpath.normpath(parsed.path.replace("\\", "/"))
            if normalized in contents:
                indexed_pages.add(normalized)
        for page_path in sorted(set(contents) - indexed_pages, key=str.casefold):
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_from_index",
                    "path": page_path,
                    "message": "Knowledge page is missing from index.md.",
                }
            )
        for message in self._link_issues("index.md", index_content):
            issues.append(
                {
                    "severity": "error",
                    "code": "broken_index_link",
                    "path": "index.md",
                    "message": message,
                }
            )
        graph = self.wiki_graph()
        for node in graph["nodes"]:
            if node["isolated"]:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "isolated_page",
                        "path": str(node["path"]),
                        "message": "Page has no links to or from another knowledge page.",
                    }
                )
            elif not node["incoming_count"]:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "no_incoming_link",
                        "path": str(node["path"]),
                        "message": "No other knowledge page links to this page.",
                    }
                )
            elif not node["outgoing_count"]:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "no_outgoing_link",
                        "path": str(node["path"]),
                        "message": "Page does not link to another knowledge page.",
                    }
                )
        issues.sort(key=lambda item: (item["severity"], item["path"].casefold(), item["code"]))
        return {
            "valid": not any(issue["severity"] == "error" for issue in issues),
            "pages_checked": len(contents),
            "issues": issues,
            "graph": graph["summary"],
        }

    @staticmethod
    def _one_line(value: object, *, limit: int = 240) -> str:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        if len(cleaned) > limit:
            return cleaned[: limit - 1].rstrip() + "…"
        return cleaned

    def append_log(
        self,
        operation: str,
        subject: str,
        *,
        status: str,
        pages: Iterable[str] = (),
        detail: str | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """Append an application-authored entry using a stable, parseable format."""

        timestamp = (occurred_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        stamp = timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")
        operation_text = self._one_line(operation.casefold(), limit=32)
        subject_text = self._one_line(subject)
        status_text = self._one_line(status.casefold(), limit=32)
        page_list = sorted(
            {self.normalize_wiki_path(page, allow_system=False) for page in pages}, key=str.casefold
        )
        entry = [f"## [{stamp}] {operation_text} | {subject_text}", "", f"- Status: {status_text}"]
        if page_list:
            entry.append(f"- Pages: {', '.join(page_list)}")
        if detail:
            entry.append(f"- Detail: {self._one_line(detail)}")
        entry_text = "\n".join(entry) + "\n\n"

        with self._write_lock:
            log_path = self.wiki_root / "log.md"
            try:
                existing = log_path.read_text(encoding="utf-8-sig")
            except FileNotFoundError:
                existing = "# Wiki Log\n\n"
            except OSError as exc:
                raise RepositoryError("Could not read wiki log.") from exc
            if existing and not existing.endswith("\n"):
                existing += "\n"
            self._atomic_write(log_path, existing + entry_text)
