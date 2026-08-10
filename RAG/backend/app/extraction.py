from __future__ import annotations

import json
import re
import threading
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .models import BoundingBox, DocumentElement, ElementType, ExtractionResult


class DocumentExtractionError(RuntimeError):
    pass


class DocumentExtractor(Protocol):
    def extract(
        self, path: Path, *, source_hash: str, artifact_dir: Path
    ) -> ExtractionResult: ...


_LABEL_MAP = {
    "title": ElementType.HEADING,
    "section_header": ElementType.HEADING,
    "paragraph": ElementType.TEXT,
    "text": ElementType.TEXT,
    "list_item": ElementType.TEXT,
    "table": ElementType.TABLE,
    "picture": ElementType.PICTURE,
    "formula": ElementType.FORMULA,
    "code": ElementType.CODE,
    "caption": ElementType.TEXT,
    "footnote": ElementType.TEXT,
}
_IGNORED_LABELS = {"page_header", "page_footer"}


class DoclingExtractor:
    """Local, ordered, element-level extraction with source provenance."""

    SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".pptx"})

    def __init__(
        self,
        *,
        do_ocr: bool = True,
        max_pages: int = 500,
        max_characters: int = 600_000,
        converter: Any | None = None,
    ):
        self.do_ocr = do_ocr
        self.max_pages = max_pages
        self.max_characters = max_characters
        self._converter = converter
        self._lock = threading.Lock()
        try:
            self.parser_version = version("docling")
        except PackageNotFoundError:
            self.parser_version = "not-installed"

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        with self._lock:
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
                    TableFormerMode,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as exc:
                raise DocumentExtractionError(
                    "Docling is not installed. Install the backend requirements."
                ) from exc

            options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    device=AcceleratorDevice.CPU, num_threads=4
                ),
                layout_options=LayoutObjectDetectionOptions(
                    engine_options=TransformersObjectDetectionEngineOptions(
                        compile_model=False
                    )
                ),
            )
            options.do_ocr = self.do_ocr
            options.do_table_structure = True
            options.table_structure_options.mode = TableFormerMode.ACCURATE
            options.images_scale = 2.0
            options.generate_picture_images = True
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                },
            )
            return self._converter

    def extract(
        self, path: Path, *, source_hash: str, artifact_dir: Path
    ) -> ExtractionResult:
        suffix = path.suffix.casefold()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise DocumentExtractionError(f"Unsupported Docling file type: {suffix}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            conversion = self._get_converter().convert(
                path, raises_on_error=True, max_num_pages=self.max_pages
            )
            document = conversion.document
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Docling could not extract {path.name}."
            ) from exc

        warnings: list[str] = []
        elements: list[DocumentElement] = []
        heading_stack: list[tuple[int, str]] = []
        total_characters = 0
        try:
            items = document.iterate_items(traverse_pictures=True)
            for index, (item, level) in enumerate(items):
                label = self._label(getattr(item, "label", None))
                if label in _IGNORED_LABELS:
                    continue
                element_type = _LABEL_MAP.get(label, ElementType.TEXT)
                text = self._item_text(item, document, element_type)
                if element_type is ElementType.PICTURE and not text:
                    text = self._caption(item, document)
                text = text.strip()
                if not text:
                    continue
                total_characters += len(text)
                if total_characters > self.max_characters:
                    raise DocumentExtractionError(
                        f"Extracted content exceeds {self.max_characters} characters."
                    )

                if element_type is ElementType.HEADING:
                    heading_level = max(int(level or 1), 1)
                    while heading_stack and heading_stack[-1][0] >= heading_level:
                        heading_stack.pop()
                    heading_stack.append((heading_level, text))

                page_number, bounding_box = self._provenance(item)
                artifact_path = None
                if element_type in {ElementType.TABLE, ElementType.PICTURE}:
                    artifact_path = self._save_image(
                        item,
                        document,
                        artifact_dir,
                        element_type,
                        index,
                        warnings,
                    )
                element_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{source_hash}:element:{index}:{label}",
                    )
                )
                elements.append(
                    DocumentElement(
                        element_id=element_id,
                        element_type=element_type,
                        text=text,
                        page_number=page_number,
                        bounding_box=bounding_box,
                        heading_path=[entry[1] for entry in heading_stack],
                        artifact_path=artifact_path,
                    )
                )
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Docling content traversal failed for {path.name}."
            ) from exc

        if not elements:
            raise DocumentExtractionError(f"No readable content found in {path.name}.")
        self._save_native_document(document, artifact_dir, warnings)
        page_count = len(getattr(document, "pages", {}) or {}) or None
        return ExtractionResult(
            parser="docling",
            parser_version=self.parser_version,
            page_count=page_count,
            elements=elements,
            warnings=warnings,
        )

    @staticmethod
    def _label(label: Any) -> str:
        return str(getattr(label, "value", label) or "text").lower()

    @staticmethod
    def _item_text(item: Any, document: Any, element_type: ElementType) -> str:
        if element_type is ElementType.TABLE:
            export = getattr(item, "export_to_markdown", None)
            if callable(export):
                try:
                    return str(export(doc=document) or "")
                except (AttributeError, TypeError, ValueError):
                    pass
        return str(getattr(item, "text", "") or "")

    @staticmethod
    def _caption(item: Any, document: Any) -> str:
        method = getattr(item, "caption_text", None)
        if not callable(method):
            return ""
        try:
            return str(method(document) or "")
        except (AttributeError, TypeError, ValueError):
            return ""

    @staticmethod
    def _provenance(item: Any) -> tuple[int | None, BoundingBox | None]:
        provenance = list(getattr(item, "prov", []) or [])
        if not provenance:
            return None, None
        first = provenance[0]
        page_number = getattr(first, "page_no", None)
        bbox = getattr(first, "bbox", None)
        box = None
        if bbox is not None:
            try:
                box = BoundingBox(
                    left=float(bbox.l),
                    top=float(bbox.t),
                    right=float(bbox.r),
                    bottom=float(bbox.b),
                )
            except (AttributeError, TypeError, ValueError):
                box = None
        return (int(page_number) if page_number is not None else None), box

    @staticmethod
    def _save_image(
        item: Any,
        document: Any,
        artifact_dir: Path,
        element_type: ElementType,
        index: int,
        warnings: list[str],
    ) -> str | None:
        get_image = getattr(item, "get_image", None)
        if not callable(get_image):
            return None
        relative = Path(f"{element_type.value}s") / f"{element_type.value}_{index:05d}.png"
        destination = artifact_dir / relative
        try:
            image = get_image(document)
            if image is None:
                return None
            if not hasattr(image, "save"):
                image = image.pil_image
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            return relative.as_posix()
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            warnings.append(f"Could not save {relative.name}: {exc}")
            return None

    @staticmethod
    def _save_native_document(
        document: Any, artifact_dir: Path, warnings: list[str]
    ) -> None:
        destination = artifact_dir / "docling_document.json"
        try:
            method = getattr(document, "save_as_json", None)
            if callable(method):
                method(destination)
                return
            export = getattr(document, "export_to_dict", None)
            if callable(export):
                destination.write_text(
                    json.dumps(export(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except (OSError, TypeError, ValueError) as exc:
            warnings.append(f"Could not save Docling JSON: {exc}")


class TextDocumentExtractor:
    SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".csv", ".json"})
    _HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, *, max_characters: int = 600_000):
        self.max_characters = max_characters

    def extract(
        self, path: Path, *, source_hash: str, artifact_dir: Path
    ) -> ExtractionResult:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise DocumentExtractionError(f"Could not read {path.name} as UTF-8.") from exc
        if not text.strip():
            raise DocumentExtractionError(f"No readable content found in {path.name}.")
        if len(text) > self.max_characters:
            raise DocumentExtractionError(
                f"Extracted content exceeds {self.max_characters} characters."
            )

        heading_stack: list[tuple[int, str]] = []
        elements: list[DocumentElement] = []
        paragraphs = re.split(r"\n\s*\n", text)
        for index, paragraph in enumerate(paragraphs):
            value = paragraph.strip()
            if not value:
                continue
            match = self._HEADING.match(value)
            element_type = ElementType.TEXT
            if match:
                level, value = len(match.group(1)), match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, value))
                element_type = ElementType.HEADING
            elements.append(
                DocumentElement(
                    element_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{source_hash}:element:{index}:text",
                        )
                    ),
                    element_type=element_type,
                    text=value,
                    heading_path=[entry[1] for entry in heading_stack],
                )
            )
        return ExtractionResult(
            parser="utf8-text",
            parser_version="1",
            elements=elements,
        )


class CompositeExtractor:
    SUPPORTED_SUFFIXES = DoclingExtractor.SUPPORTED_SUFFIXES | TextDocumentExtractor.SUPPORTED_SUFFIXES

    def __init__(self, docling: DoclingExtractor, text: TextDocumentExtractor):
        self.docling = docling
        self.text = text

    def extract(
        self, path: Path, *, source_hash: str, artifact_dir: Path
    ) -> ExtractionResult:
        if path.suffix.casefold() in DoclingExtractor.SUPPORTED_SUFFIXES:
            return self.docling.extract(
                path, source_hash=source_hash, artifact_dir=artifact_dir
            )
        if path.suffix.casefold() in TextDocumentExtractor.SUPPORTED_SUFFIXES:
            return self.text.extract(
                path, source_hash=source_hash, artifact_dir=artifact_dir
            )
        raise DocumentExtractionError(f"Unsupported file type: {path.suffix or '(none)'}")

