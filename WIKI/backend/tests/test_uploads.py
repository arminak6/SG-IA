from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.agent import IngestionResult, parse_ingestion_prompt
from backend.app.config import BedrockSettings
from backend.app.repository import UploadValidationError, WikiRepository
from backend.app.service import WikiService


def wiki_page(source_path: str) -> str:
    return f"""---
title: Shared upload
page_type: source
updated: 2026-09-03
sources:
  - {source_path}
---

# Shared upload

Shared knowledge.

## Sources

- {source_path}
"""


class CommittingUploadAgent:
    def __init__(self, repository: WikiRepository) -> None:
        self.repository = repository
        self.sources: list[str] = []

    def ingest(self, prompt: str) -> IngestionResult:
        source_path = parse_ingestion_prompt(prompt)
        self.sources.append(source_path)
        pages = self.repository.commit_ingestion(
            source_path,
            {"sources/shared-upload.md": wiki_page(source_path)},
        )
        return IngestionResult(
            source_path=source_path,
            prompt=prompt,
            pages_written=tuple(pages),
            message="Integrated shared knowledge.",
            usage={},
        )


class SharedUploadTests(unittest.TestCase):
    def test_repository_stores_safe_content_addressed_immutable_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            repository = WikiRepository(backend_root)

            first, first_duplicate = repository.save_uploaded_source(
                "../../Quarter:Plan.txt",
                b"Shared knowledge",
            )
            second, second_duplicate = repository.save_uploaded_source(
                "renamed.txt",
                b"Shared knowledge",
            )

            self.assertFalse(first_duplicate)
            self.assertTrue(second_duplicate)
            self.assertEqual(first.relative_path, second.relative_path)
            self.assertTrue(first.relative_path.startswith("uploads/"))
            self.assertTrue(first.relative_path.endswith("/Quarter_Plan.txt"))
            self.assertEqual(
                (backend_root / "raw" / first.relative_path).read_bytes(),
                b"Shared knowledge",
            )
            self.assertEqual(len(repository.list_raw_documents()), 1)

    def test_repository_rejects_invalid_uploads_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = WikiRepository(
                Path(temp_dir) / "backend",
                max_source_bytes=4,
            )

            with self.assertRaisesRegex(UploadValidationError, "empty"):
                repository.save_uploaded_source("empty.txt", b"")
            with self.assertRaisesRegex(UploadValidationError, "byte limit"):
                repository.save_uploaded_source("large.txt", b"12345")
            with self.assertRaisesRegex(UploadValidationError, "Unsupported"):
                repository.save_uploaded_source("program.exe", b"1234")
            with self.assertRaisesRegex(UploadValidationError, "UTF-8"):
                repository.save_uploaded_source("invalid.txt", b"\xff")

    def test_service_uploads_ingests_and_reuses_shared_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            repository = WikiRepository(backend_root)
            agent = CommittingUploadAgent(repository)
            settings = BedrockSettings(
                project_root=project_root,
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            service = WikiService(settings, repository=repository, agent=agent)

            first = service.upload_document("shared.txt", b"Shared knowledge")
            second = service.upload_document("another-name.txt", b"Shared knowledge")

            self.assertFalse(first["duplicate"])
            self.assertEqual(first["update"]["summary"]["processed"], 1)
            self.assertEqual(first["document"]["status"], "Ingested")
            self.assertTrue(second["duplicate"])
            self.assertEqual(second["update"]["summary"]["skipped"], 1)
            self.assertEqual(len(agent.sources), 1)


if __name__ == "__main__":
    unittest.main()
