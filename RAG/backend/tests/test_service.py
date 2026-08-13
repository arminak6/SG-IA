from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.chunking import StructureAwareChunker
from app.confidence import ConfidenceEvaluation
from app.generation import GeneratedAnswer
from app.models import (
    DocumentElement,
    ElementType,
    ExtractionResult,
    JobStatus,
    RagChunk,
    SearchHit,
    SearchResponse,
)
from app.repository import LocalRepository
from app.service import RagService, UploadValidationError


class FakeExtractor:
    SUPPORTED_SUFFIXES = frozenset({".txt"})

    def extract(self, path: Path, *, source_hash: str, artifact_dir: Path) -> ExtractionResult:
        return ExtractionResult(
            parser="fake",
            parser_version="1",
            elements=[
                DocumentElement(
                    element_id="element-1",
                    element_type=ElementType.TEXT,
                    text=path.read_text(encoding="utf-8"),
                    page_number=1,
                    heading_path=["General"],
                )
            ],
        )


class FakeEmbeddings:
    model_id = "fake-embeddings"
    dimension = 256

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 255 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0] + [0.0] * 255


class FakeGenerator:
    model_id = "fake-generation"

    def generate(self, question, evidence) -> GeneratedAnswer:
        return GeneratedAnswer(
            status="answered",
            answer="Two reviewers are required.",
            evidence_ids=("E1",),
            usage={"inputTokens": 20, "outputTokens": 5, "totalTokens": 25},
            stop_reason="tool_use",
            attempts=1,
        )


class FakeConfidenceEvaluator:
    model_id = "fake-confidence"

    def __init__(self) -> None:
        self.calls = []

    def evaluate(
        self,
        question,
        result,
        evidence,
        *,
        evidence_coverage_ratio,
        retrieval_attempts,
    ) -> ConfidenceEvaluation:
        self.calls.append(
            {
                "question": question,
                "result": result,
                "evidence": evidence,
                "coverage": evidence_coverage_ratio,
                "retrieval_attempts": retrieval_attempts,
            }
        )
        return ConfidenceEvaluation(
            score=8.7,
            usage={"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
            claim_support=0.9,
            question_coverage=0.8,
            source_consistency=1.0,
            evidence_quality=0.8,
            abstention_score=1.0,
            has_unsupported_material_claim=False,
            has_unexplained_conflict=False,
        )


class BrokenConfidenceEvaluator:
    model_id = "broken-confidence"

    def evaluate(self, *_args, **_kwargs):
        raise RuntimeError("private verifier failure")


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: dict[str, list[RagChunk]] = {}
        self.deleted: list[str] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_document(self, document_id: str, chunks: Sequence[RagChunk], vectors: Sequence[Sequence[float]]) -> None:
        self.chunks[document_id] = list(chunks)

    def count_document(self, document_id: str) -> int:
        return len(self.chunks.get(document_id, []))

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)
        self.chunks.pop(document_id, None)

    def search(self, vector: Sequence[float], *, limit: int, document_ids: list[str] | None = None) -> list[SearchHit]:
        allowed = set(document_ids or self.chunks)
        chunks = [chunk for doc_id, values in self.chunks.items() if doc_id in allowed for chunk in values]
        return [
            SearchHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                title=chunk.title,
                score=0.93,
                text=chunk.text,
                page_numbers=chunk.page_numbers,
                heading_path=chunk.heading_path,
                content_types=[item.value for item in chunk.content_types],
            )
            for chunk in chunks[:limit]
        ]

    def neighbors(self, hits: Sequence[SearchHit], *, window: int) -> list[SearchHit]:
        if window <= 0:
            return []
        by_document = {
            document_id: {chunk.ordinal: chunk for chunk in chunks}
            for document_id, chunks in self.chunks.items()
        }
        result: list[SearchHit] = []
        seen = {hit.chunk_id for hit in hits}
        for hit in hits:
            ordinal = int(hit.metadata.get("ordinal", 0))
            for neighbor_ordinal in range(max(0, ordinal - window), ordinal + window + 1):
                chunk = by_document.get(hit.document_id, {}).get(neighbor_ordinal)
                if chunk is None or chunk.chunk_id in seen or chunk.heading_path != hit.heading_path:
                    continue
                result.append(
                    SearchHit(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        title=chunk.title,
                        score=0.9,
                        text=chunk.text,
                        page_numbers=chunk.page_numbers,
                        heading_path=chunk.heading_path,
                        content_types=[item.value for item in chunk.content_types],
                        metadata={
                            "ordinal": chunk.ordinal,
                            "retrieval_origin": "neighbor",
                            "neighbor_of_chunk_ids": [hit.chunk_id],
                        },
                    )
                )
                seen.add(chunk.chunk_id)
        return result

    def health(self) -> bool:
        return True


def build_service(
    settings, *, confidence_evaluator=None
) -> tuple[RagService, FakeVectorStore]:
    vector_store = FakeVectorStore()
    service = RagService(
        settings=settings,
        repository=LocalRepository(settings),
        extractor=FakeExtractor(),
        chunker=StructureAwareChunker(max_tokens=100, overlap_tokens=10),
        embeddings=FakeEmbeddings(),
        generator=FakeGenerator(),
        vector_store=vector_store,
        confidence_evaluator=confidence_evaluator,
    )
    return service, vector_store


def test_ingestion_is_verified_and_searchable(settings) -> None:
    service, vector_store = build_service(settings)
    accepted = service.accept_upload(
        filename="handbook.txt",
        content=b"The approval procedure requires two reviewers.",
        title="Handbook",
        media_type="text/plain",
    )

    service.ingest(accepted.job.job_id)

    job = service.repository.get_job(accepted.job.job_id)
    assert job is not None and job.status is JobStatus.COMPLETED
    document = service.repository.get_document(job.document_id)
    assert document is not None
    assert document.chunk_count == vector_store.count_document(job.document_id) == 1
    result = service.search("How many reviewers?", top_k=5, document_ids=None)
    assert result.hits[0].filename == "handbook.txt"
    assert result.hits[0].page_numbers == [1]

    answer = service.chat(
        "How many reviewers?",
        top_k=5,
        document_ids=None,
        session_id="session-1",
    )
    assert answer.approach == "rag"
    assert answer.status == "answered"
    assert answer.answer == "Two reviewers are required."
    assert answer.citations[0].source_path == "handbook.txt"
    assert answer.citations[0].page_numbers == [1]
    assert answer.debug.cited_chunk_ids == [answer.citations[0].chunk_id]
    assert answer.debug.session_id == "session-1"
    assert answer.debug.retrieval_strategy.endswith("v1.2")
    assert answer.debug.candidate_pool_size == 24
    assert answer.debug.final_context_count == 1


def test_chat_adds_advisory_confidence_score_and_verifier_usage(settings) -> None:
    confidence = FakeConfidenceEvaluator()
    service, _ = build_service(settings, confidence_evaluator=confidence)
    accepted = service.accept_upload(
        filename="handbook.txt",
        content=b"The approval procedure requires two reviewers.",
        title="Handbook",
        media_type="text/plain",
    )
    service.ingest(accepted.job.job_id)

    answer = service.chat(
        "How many reviewers?",
        top_k=8,
        document_ids=None,
        session_id="confidence-session",
    )

    assert answer.status == "answered"
    assert answer.confidence_score == 8.7
    assert answer.usage["totalTokens"] == 39
    assert answer.debug.confidence.enabled is True
    assert answer.debug.confidence.verification_available is True
    assert answer.debug.confidence.model_id == "fake-confidence"
    assert answer.debug.confidence.components["claim_support"] == 0.9
    assert answer.timings.verification_ms >= 0
    assert confidence.calls[0]["question"] == "How many reviewers?"


def test_confidence_failure_keeps_grounded_answer_available(settings) -> None:
    service, _ = build_service(
        settings,
        confidence_evaluator=BrokenConfidenceEvaluator(),
    )
    accepted = service.accept_upload(
        filename="handbook.txt",
        content=b"The approval procedure requires two reviewers.",
        title="Handbook",
        media_type="text/plain",
    )
    service.ingest(accepted.job.job_id)

    answer = service.chat(
        "How many reviewers?",
        top_k=8,
        document_ids=None,
        session_id="broken-confidence-session",
    )

    assert answer.status == "answered"
    assert answer.answer == "Two reviewers are required."
    assert answer.confidence_score is None
    assert answer.debug.confidence.verification_available is False
    assert answer.debug.confidence.reasons == ["verification_unavailable"]


def test_duplicate_content_reuses_indexed_document(settings) -> None:
    service, _ = build_service(settings)
    first = service.accept_upload(
        filename="first.txt", content=b"Same content", title=None, media_type="text/plain"
    )
    service.ingest(first.job.job_id)

    duplicate = service.accept_upload(
        filename="renamed.txt", content=b"Same content", title=None, media_type="text/plain"
    )

    assert duplicate.duplicate is True
    assert duplicate.schedule_ingestion is False
    assert duplicate.job.document_id == first.job.document_id
    assert len(service.repository.list_documents()) == 1


def test_unsupported_file_is_rejected(settings) -> None:
    service, _ = build_service(settings)

    try:
        service.accept_upload(
            filename="script.exe", content=b"data", title=None, media_type=None
        )
    except UploadValidationError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("unsupported file should fail")


def test_chat_retries_retrieval_once_when_numeric_facets_are_missing(
    settings, monkeypatch
) -> None:
    service, _ = build_service(settings)
    calls: list[str] = []

    def fake_search(query: str, *, top_k: int, document_ids: list[str] | None):
        calls.append(query)
        text = (
            "Milestones in 2005, 2010 and 2020."
            if len(calls) == 2
            else "Milestones in 2010 and 2020."
        )
        return SearchResponse(
            query=query,
            hits=[
                SearchHit(
                    chunk_id=f"chunk-{len(calls)}",
                    document_id="doc-1",
                    filename="history.txt",
                    title="History",
                    score=0.8,
                    text=text,
                    heading_path=["Milestones"],
                    metadata={"ordinal": len(calls)},
                )
            ],
            latency_ms=1,
            embedding_model_id="fake-embeddings",
        )

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr(service.vector_store, "neighbors", lambda hits, window: [])

    retrieval = service._retrieve_for_chat(
        question="List the milestones in 2005, 2010 and 2020.",
        final_top_k=5,
        document_ids=None,
    )

    assert retrieval.attempts == 2
    assert retrieval.coverage.sufficient is True
    assert "2005" in calls[1]
