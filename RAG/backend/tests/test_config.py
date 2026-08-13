from __future__ import annotations

import json

from app.config import Settings


def test_model_registry_is_loaded_and_environment_can_override(tmp_path, monkeypatch) -> None:
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extraction": {
                    "engine": "docling",
                    "layout_model_id": "layout-test",
                    "ocr_model_id": "ocr-test",
                    "table_structure_model_id": "table-test",
                    "ocr_enabled": False,
                },
                "embedding": {
                    "model_id": "amazon.titan-embed-text-v2:0",
                    "dimensions": 256,
                },
                "generation": {
                    "model_id": "generation-from-json",
                    "temperature": 0.2,
                    "max_output_tokens": 900,
                },
                "verification": {
                    "enabled": True,
                    "model_id": "confidence-from-json",
                    "max_output_tokens": 400,
                    "max_evidence_characters": 8_000,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_MODEL_CONFIG_PATH", str(model_config))
    monkeypatch.setenv("BEDROCK_GENERATION_MODEL_ID", "generation-from-env")
    monkeypatch.delenv("BEDROCK_EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("DOCLING_DO_OCR", raising=False)

    settings = Settings.from_env()

    assert settings.embedding_dimensions == 256
    assert settings.generation_model_id == "generation-from-env"
    assert settings.generation_max_output_tokens == 900
    assert settings.confidence_enabled is True
    assert settings.confidence_model_id == "confidence-from-json"
    assert settings.confidence_max_output_tokens == 400
    assert settings.confidence_max_evidence_characters == 8_000
    assert settings.layout_model_id == "layout-test"
    assert settings.docling_do_ocr is False
