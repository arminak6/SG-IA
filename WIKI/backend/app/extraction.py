"""Local structured extraction for binary office documents."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol


class DocumentExtractionError(RuntimeError):
    """Raised when a supported binary document cannot be converted safely."""


class DocumentExtractor(Protocol):
    """Interface used by the repository to convert immutable raw documents."""

    def extract(self, file_path: Path, *, source_path: str) -> str:
        """Return normalized Markdown without modifying ``file_path``."""


class DoclingDocumentExtractor:
    """Convert PDF, DOCX, and PPTX sources to provenance-marked Markdown.

    Docling is imported and its converter is initialized only when a binary
    document is first read. This keeps text-only API operations and the test
    suite lightweight while ensuring conversion stays local and in memory.
    """

    SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".pptx"})
    PAGE_BREAK = "\n\n<!-- docling-page-break -->\n\n"

    def __init__(
        self,
        *,
        converter: Any | None = None,
        max_extracted_characters: int = 600_000,
        max_pages: int = 500,
    ) -> None:
        self._converter = converter
        self.max_extracted_characters = max_extracted_characters
        self.max_pages = max_pages
        self._converter_lock = threading.Lock()

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        with self._converter_lock:
            if self._converter is not None:
                return self._converter
            try:
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.object_detection_engine_options import (
                    TransformersObjectDetectionEngineOptions,
                )
                from docling.datamodel.pipeline_options import (
                    LayoutObjectDetectionOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as exc:  # pragma: no cover - depends on deployment
                raise DocumentExtractionError(
                    "Docling is not installed; install backend/requirements.txt."
                ) from exc

            # Docling enables torch.compile by default. On CPU-only Windows
            # machines that can require the optional Visual C++ compiler. Eager
            # CPU inference is slower but works without external build tools and
            # gives this local application a predictable deployment baseline.
            pipeline_options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    device=AcceleratorDevice.CPU,
                    num_threads=4,
                ),
                layout_options=LayoutObjectDetectionOptions(
                    engine_options=TransformersObjectDetectionEngineOptions(
                        compile_model=False
                    )
                ),
            )
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                },
            )
            return self._converter

    @classmethod
    def _mark_pages(cls, markdown: str, suffix: str) -> str:
        if suffix not in {".pdf", ".pptx"}:
            return markdown.strip()

        label = "page" if suffix == ".pdf" else "slide"
        sections = [section.strip() for section in markdown.split(cls.PAGE_BREAK)]
        sections = [section for section in sections if section]
        return "\n\n".join(
            f"<!-- Source {label} {number} -->\n\n{section}"
            for number, section in enumerate(sections, start=1)
        )

    def extract(self, file_path: Path, *, source_path: str) -> str:
        suffix = file_path.suffix.casefold()
        if suffix not in self.SUPPORTED_SUFFIXES:
            suffix_label = suffix or "(none)"
            raise DocumentExtractionError(
                f"Unsupported Docling source type: {suffix_label}."
            )

        try:
            result = self._get_converter().convert(
                file_path,
                raises_on_error=True,
                max_num_pages=self.max_pages,
            )
            document = result.document
            markdown = document.export_to_markdown(
                page_break_placeholder=self.PAGE_BREAK,
                traverse_pictures=True,
            )
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Docling could not extract content from {source_path}."
            ) from exc

        marked = self._mark_pages(str(markdown), suffix)
        if not marked.strip():
            raise DocumentExtractionError(
                f"Docling found no readable content in {source_path}."
            )

        content = (
            f"<!-- Extracted locally with Docling from {source_path}; "
            "the original file is authoritative. -->\n\n"
            f"{marked.strip()}\n"
        )
        if len(content) > self.max_extracted_characters:
            raise DocumentExtractionError(
                f"Extracted content for {source_path} is too large "
                f"({len(content)} characters; limit {self.max_extracted_characters})."
            )
        return content
