from pathlib import Path

import pytest
from app.extraction import DocumentExtractionError, TextDocumentExtractor
from app.models import ElementType


def test_markdown_extraction_preserves_heading_paths(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nWelcome.\n\n## Procedure\n\nDo this.", encoding="utf-8")

    result = TextDocumentExtractor().extract(
        source, source_hash="hash", artifact_dir=tmp_path / "artifacts"
    )

    assert [item.element_type for item in result.elements] == [
        ElementType.HEADING,
        ElementType.TEXT,
        ElementType.HEADING,
        ElementType.TEXT,
    ]
    assert result.elements[-1].heading_path == ["Guide", "Procedure"]


def test_empty_text_document_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("   ", encoding="utf-8")

    with pytest.raises(DocumentExtractionError, match="No readable content"):
        TextDocumentExtractor().extract(
            source, source_hash="hash", artifact_dir=tmp_path / "artifacts"
        )

