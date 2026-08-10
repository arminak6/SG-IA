from __future__ import annotations

from pathlib import Path

import pytest
from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        qdrant_url="http://qdrant.test:6333",
        qdrant_collection="test_chunks",
        aws_region="eu-central-1",
        embedding_model_id="fake-embeddings",
        embedding_dimensions=256,
        chunk_max_tokens=100,
        chunk_overlap_tokens=10,
        max_upload_bytes=1_000_000,
        max_document_pages=50,
        max_extracted_characters=100_000,
        docling_do_ocr=False,
    )

