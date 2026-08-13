from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from evaluate_rag import citation_metrics, collect_rag_evidence, validate_benchmark


def test_validate_shared_benchmark_shape() -> None:
    value = {
        "cases": [
            {
                "id": "case-1",
                "question": "Domanda?",
                "expected_status": "answered",
                "ground_truth_answer": "Risposta.",
                "required_answer_points": ["Punto."],
                "sources": [],
            }
        ]
    }
    assert validate_benchmark(value) is value


def test_citation_metrics_compare_normalized_basenames() -> None:
    case = {"sources": [{"source_path": "raw/cartella/Manuale & Guida (IT).pdf"}]}
    response = {"citations": [{"source_path": "Manuale _ Guida _IT_.pdf"}]}
    metrics = citation_metrics(case, response)
    assert metrics["expected_source_recall"] == 1.0
    assert metrics["expected_source_precision"] == 1.0


def test_collect_rag_evidence_includes_only_cited_chunks() -> None:
    response = {
        "citations": [
            {"evidence_id": "E1", "chunk_id": "chunk-1", "source_path": "manuale.pdf"}
        ],
        "debug": {
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "filename": "manuale.pdf",
                    "text": "contenuto citato",
                    "page_numbers": [2],
                    "heading_path": ["Sezione"],
                },
                {
                    "chunk_id": "chunk-2",
                    "filename": "altro.pdf",
                    "text": "contenuto non citato",
                },
            ]
        },
    }
    chunks, metadata = collect_rag_evidence(
        response, max_chars_per_chunk=100, max_total_chars=1000
    )
    assert [item["chunk_id"] for item in chunks] == ["chunk-1"]
    assert chunks[0]["evidence_id"] == "E1"
    assert metadata[0]["available"] is True
