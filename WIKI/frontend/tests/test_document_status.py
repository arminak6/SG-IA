import tempfile
import unittest
from pathlib import Path

from frontend.document_status import build_ingestion_prompt, scan_documents


class DocumentStatusTests(unittest.TestCase):
    def test_build_ingestion_prompt_uses_backend_relative_path(self) -> None:
        prompt = build_ingestion_prompt(r"research\qdrant-article.md")
        self.assertEqual(
            prompt,
            "Ingest raw/research/qdrant-article.md into the wiki.",
        )

    def test_scan_marks_unreferenced_document_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            raw_dir.mkdir()
            wiki_dir.mkdir()
            (raw_dir / "article.txt").write_text("Source", encoding="utf-8")

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].relative_path, "article.txt")
            self.assertEqual(documents[0].status, "Pending")

    def test_scan_marks_exactly_referenced_document_as_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            (raw_dir / "research").mkdir(parents=True)
            wiki_dir.mkdir()
            (raw_dir / "research" / "article.md").write_text(
                "Source", encoding="utf-8"
            )
            (wiki_dir / "article-summary.md").write_text(
                "Source: raw/research/article.md", encoding="utf-8"
            )

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].status, "Ingested")

    def test_scan_does_not_match_a_longer_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            raw_dir.mkdir()
            wiki_dir.mkdir()
            (raw_dir / "article.md").write_text("Source", encoding="utf-8")
            (wiki_dir / "article-summary.md").write_text(
                "Source: raw/article.md.backup", encoding="utf-8"
            )

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual(documents[0].status, "Pending")

    def test_system_log_reference_does_not_mark_failed_source_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            raw_dir.mkdir()
            wiki_dir.mkdir()
            (raw_dir / "article.md").write_text("Source", encoding="utf-8")
            (wiki_dir / "log.md").write_text(
                "## ingest | raw/article.md\n\n- Status: failed\n", encoding="utf-8"
            )
            (wiki_dir / "index.md").write_text(
                "Pending: raw/article.md\n", encoding="utf-8"
            )

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual(documents[0].status, "Pending")

    def test_scan_skips_hidden_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            raw_dir.mkdir()
            wiki_dir.mkdir()
            (raw_dir / ".gitkeep").write_text("", encoding="utf-8")
            (raw_dir / "visible.txt").write_text("Source", encoding="utf-8")

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual([item.relative_path for item in documents], ["visible.txt"])

    def test_scan_includes_docling_formats_and_skips_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            wiki_dir = root / "wiki"
            raw_dir.mkdir()
            wiki_dir.mkdir()
            for filename in ("report.pdf", "policy.docx", "slides.pptx", "archive.zip"):
                (raw_dir / filename).write_bytes(b"source")

            documents = scan_documents(raw_dir, wiki_dir)

            self.assertEqual(
                [item.relative_path for item in documents],
                ["policy.docx", "report.pdf", "slides.pptx"],
            )


if __name__ == "__main__":
    unittest.main()
