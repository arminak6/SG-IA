from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app.extraction import DoclingDocumentExtractor, DocumentExtractionError
from backend.app.repository import SourceReadError, WikiRepository


class FakeDoclingDocument:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.export_options: dict[str, object] | None = None

    def export_to_markdown(self, **kwargs) -> str:
        self.export_options = kwargs
        return self.markdown


class FakeDoclingConverter:
    def __init__(self, markdown: str) -> None:
        self.document = FakeDoclingDocument(markdown)
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def convert(self, path: Path, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(document=self.document)


class RecordingExtractor:
    def __init__(self, content: str = "# Extracted\n\nUseful knowledge.\n") -> None:
        self.content = content
        self.calls: list[tuple[Path, str]] = []

    def extract(self, file_path: Path, *, source_path: str) -> str:
        self.calls.append((file_path, source_path))
        return self.content


class FailingExtractor:
    def extract(self, file_path: Path, *, source_path: str) -> str:
        raise DocumentExtractionError(f"Could not extract {source_path}.")


class ExtractionTests(unittest.TestCase):
    def test_pdf_extraction_adds_provenance_and_page_markers_without_writes(self) -> None:
        markdown = (
            "# First page\n\nAlpha"
            f"{DoclingDocumentExtractor.PAGE_BREAK}"
            "# Second page\n\nBeta"
        )
        converter = FakeDoclingConverter(markdown)
        extractor = DoclingDocumentExtractor(converter=converter)

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "report.pdf"
            source.write_bytes(b"unchanged-pdf")
            before = set(source.parent.iterdir())

            content = extractor.extract(source, source_path="raw/report.pdf")

            self.assertEqual(source.read_bytes(), b"unchanged-pdf")
            self.assertEqual(set(source.parent.iterdir()), before)

        self.assertIn("Docling from raw/report.pdf", content)
        self.assertIn("<!-- Source page 1 -->", content)
        self.assertIn("<!-- Source page 2 -->", content)
        self.assertEqual(converter.calls[0][1]["max_num_pages"], 500)
        self.assertTrue(converter.calls[0][1]["raises_on_error"])
        self.assertTrue(converter.document.export_options["traverse_pictures"])

    def test_pptx_uses_slide_markers_and_docx_preserves_document_flow(self) -> None:
        for suffix, expected_marker in (
            (".pptx", "<!-- Source slide 1 -->"),
            (".docx", None),
        ):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / f"source{suffix}"
                source.write_bytes(b"office-document")
                converter = FakeDoclingConverter("# Heading\n\nBody")
                extractor = DoclingDocumentExtractor(converter=converter)

                content = extractor.extract(source, source_path=f"raw/source{suffix}")

                if expected_marker:
                    self.assertIn(expected_marker, content)
                else:
                    self.assertNotIn("<!-- Source page", content)
                    self.assertNotIn("<!-- Source slide", content)

    def test_empty_and_oversized_extraction_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "report.pdf"
            source.write_bytes(b"pdf")

            empty = DoclingDocumentExtractor(converter=FakeDoclingConverter("  "))
            with self.assertRaisesRegex(DocumentExtractionError, "no readable content"):
                empty.extract(source, source_path="raw/report.pdf")

            oversized = DoclingDocumentExtractor(
                converter=FakeDoclingConverter("x" * 100),
                max_extracted_characters=50,
            )
            with self.assertRaisesRegex(DocumentExtractionError, "too large"):
                oversized.extract(source, source_path="raw/report.pdf")

    def test_repository_routes_pdf_docx_and_pptx_through_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            for suffix in (".pdf", ".docx", ".pptx"):
                (raw_root / f"source{suffix}").write_bytes(b"original")
            extractor = RecordingExtractor()
            repository = WikiRepository(backend_root, document_extractor=extractor)

            for suffix in (".pdf", ".docx", ".pptx"):
                content = repository.read_raw(f"raw/source{suffix}")
                self.assertIn("Useful knowledge", content)
                self.assertEqual((raw_root / f"source{suffix}").read_bytes(), b"original")

            self.assertEqual(
                [source_path for _, source_path in extractor.calls],
                ["raw/source.pdf", "raw/source.docx", "raw/source.pptx"],
            )

    def test_repository_surfaces_safe_extraction_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "report.pdf").write_bytes(b"pdf")
            repository = WikiRepository(backend_root, document_extractor=FailingExtractor())

            with self.assertRaisesRegex(SourceReadError, "raw/report.pdf"):
                repository.read_raw("raw/report.pdf")

    def test_repository_lists_only_supported_source_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "report.pdf").write_bytes(b"pdf")
            (raw_root / "archive.zip").write_bytes(b"zip")
            repository = WikiRepository(backend_root)

            self.assertEqual(
                [document.relative_path for document in repository.list_raw_documents()],
                ["report.pdf"],
            )


if __name__ == "__main__":
    unittest.main()
