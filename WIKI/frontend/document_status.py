"""Document discovery helpers for the Streamlit frontend.

The backend will eventually own ingestion state. Until then, the frontend uses
source provenance already written into wiki Markdown files: a raw document is
considered ingested when an exact ``raw/<relative-path>`` reference exists in
the wiki.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SOURCE_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".json", ".csv", ".pdf", ".docx", ".pptx"}
)


@dataclass(frozen=True)
class DocumentStatus:
    """Display information for one file in the raw source directory."""

    relative_path: str
    size_bytes: int
    modified_timestamp: float
    is_ingested: bool

    @property
    def status(self) -> str:
        return "Ingested" if self.is_ingested else "Pending"


def build_ingestion_prompt(relative_path: str) -> str:
    """Build the backend instruction for a raw document."""

    normalized_path = relative_path.replace("\\", "/").lstrip("/")
    return f"Ingest raw/{normalized_path} into the wiki."


def scan_documents(raw_dir: Path, wiki_dir: Path) -> list[DocumentStatus]:
    """Return raw files and infer their current ingestion status.

    Hidden files and files inside hidden directories are omitted from the
    document list. Wiki pages are expected to be Markdown and may cite a source
    anywhere using its path relative to ``backend``, for example
    ``raw/research/article.md``.
    """

    if not raw_dir.is_dir():
        return []

    wiki_text = _read_wiki_markdown(wiki_dir)
    documents: list[DocumentStatus] = []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(raw_dir)
        if _is_hidden(relative):
            continue
        if path.suffix.casefold() not in SUPPORTED_SOURCE_SUFFIXES:
            continue

        relative_path = relative.as_posix()
        source_reference = f"raw/{relative_path}".casefold()
        stat = path.stat()
        documents.append(
            DocumentStatus(
                relative_path=relative_path,
                size_bytes=stat.st_size,
                modified_timestamp=stat.st_mtime,
                is_ingested=_contains_source_reference(wiki_text, source_reference),
            )
        )

    return sorted(documents, key=lambda item: item.relative_path.casefold())


def _read_wiki_markdown(wiki_dir: Path) -> str:
    if not wiki_dir.is_dir():
        return ""

    contents: list[str] = []
    markdown_files = list(wiki_dir.rglob("*.md")) + list(
        wiki_dir.rglob("*.markdown")
    )
    for path in markdown_files:
        relative_path = path.relative_to(wiki_dir).as_posix().casefold()
        if relative_path in {"index.md", "log.md"}:
            # These system pages can mention attempted/failed sources. Only
            # provenance on an actual content page proves successful ingestion.
            continue
        try:
            contents.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            # A single temporarily unavailable page should not hide every raw
            # source from the sidebar.
            continue

    return "\n".join(contents).replace("\\", "/").casefold()


def _is_hidden(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)


def _contains_source_reference(wiki_text: str, source_reference: str) -> bool:
    """Match a complete source path instead of a longer filename prefix."""

    boundary_chars = r"\w./-"
    pattern = rf"(?<![{boundary_chars}]){re.escape(source_reference)}(?![{boundary_chars}])"
    return re.search(pattern, wiki_text) is not None
