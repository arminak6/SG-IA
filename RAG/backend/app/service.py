from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass

from .chunking import StructureAwareChunker
from .config import Settings
from .embeddings import EmbeddingProvider
from .extraction import CompositeExtractor, DocumentExtractionError
from .generation import AnswerGenerator
from .models import (
    ChatDebug,
    ChatResponse,
    ChatTimings,
    DocumentRecord,
    IngestionJob,
    JobStatus,
    RagCitation,
    SearchResponse,
)
from .repository import LocalRepository
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AcceptedUpload:
    job: IngestionJob
    duplicate: bool
    schedule_ingestion: bool


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: LocalRepository,
        extractor: CompositeExtractor,
        chunker: StructureAwareChunker,
        embeddings: EmbeddingProvider,
        generator: AnswerGenerator,
        vector_store: VectorStore,
    ):
        self.settings = settings
        self.repository = repository
        self.extractor = extractor
        self.chunker = chunker
        self.embeddings = embeddings
        self.generator = generator
        self.vector_store = vector_store
        self._accept_lock = threading.Lock()

    def accept_upload(
        self,
        *,
        filename: str,
        content: bytes,
        title: str | None,
        media_type: str | None,
    ) -> AcceptedUpload:
        safe_filename = self.repository.safe_filename(filename)
        suffix = self._validate_upload(safe_filename, content)
        source_hash = hashlib.sha256(content).hexdigest()
        with self._accept_lock:
            existing = self.repository.find_document_by_hash(source_hash)
            if existing is not None:
                job = IngestionJob(
                    job_id=str(uuid.uuid4()),
                    document_id=existing.document_id,
                    filename=existing.filename,
                    source_hash=source_hash,
                    status=JobStatus.COMPLETED,
                    stage="already_indexed",
                    duplicate=True,
                    chunk_count=existing.chunk_count,
                )
                self.repository.save_job(job)
                return AcceptedUpload(job=job, duplicate=True, schedule_ingestion=False)

            active = self.repository.find_active_job_by_hash(source_hash)
            if active is not None:
                active.duplicate = True
                self.repository.save_job(active)
                return AcceptedUpload(job=active, duplicate=True, schedule_ingestion=False)

            document_id = str(uuid.uuid4())
            job = IngestionJob(
                job_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=safe_filename,
                source_hash=source_hash,
            )
            self.repository.save_upload(document_id, safe_filename, content)
            self.repository.save_job(job)
            # Store request-only metadata without exposing it as an indexed record.
            pending = self.repository.artifact_dir(document_id) / "upload_metadata.json"
            pending.write_text(
                json.dumps(
                    {
                        "title": self._clean_title(title, safe_filename),
                        "media_type": media_type
                        or mimetypes.guess_type(safe_filename)[0]
                        or "",
                        "suffix": suffix,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return AcceptedUpload(job=job, duplicate=False, schedule_ingestion=True)

    def ingest(self, job_id: str) -> None:
        job = self.repository.get_job(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return
        try:
            job.status = JobStatus.PROCESSING
            job.stage = "extracting"
            self.repository.save_job(job)
            upload_path = self.repository.upload_path(job.document_id)
            metadata = self._read_upload_metadata(job.document_id)
            extraction = self.extractor.extract(
                upload_path,
                source_hash=job.source_hash,
                artifact_dir=self.repository.artifact_dir(job.document_id),
            )
            self.repository.save_artifact_model(
                job.document_id, "extraction_manifest.json", extraction
            )

            job.stage = "chunking"
            self.repository.save_job(job)
            chunks = self.chunker.chunk(
                document_id=job.document_id,
                source_hash=job.source_hash,
                filename=job.filename,
                title=metadata["title"],
                elements=extraction.elements,
            )
            if not chunks:
                raise DocumentExtractionError("Extraction produced no retrievable chunks.")

            job.stage = "embedding"
            self.repository.save_job(job)
            vectors = self.embeddings.embed_texts(
                [chunk.embedding_text for chunk in chunks]
            )

            job.stage = "indexing"
            self.repository.save_job(job)
            self.vector_store.upsert_document(job.document_id, chunks, vectors)
            indexed_count = self.vector_store.count_document(job.document_id)
            if indexed_count != len(chunks):
                raise RuntimeError(
                    f"Qdrant verification found {indexed_count} of {len(chunks)} chunks."
                )

            record = DocumentRecord(
                document_id=job.document_id,
                filename=job.filename,
                title=metadata["title"],
                source_hash=job.source_hash,
                media_type=metadata["media_type"] or None,
                created_at=job.created_at,
                parser=extraction.parser,
                parser_version=extraction.parser_version,
                page_count=extraction.page_count,
                element_count=len(extraction.elements),
                chunk_count=len(chunks),
                embedding_model_id=self.embeddings.model_id,
                embedding_dimensions=self.embeddings.dimension,
                warnings=extraction.warnings,
            )
            self.repository.save_document(record)
            job.status = JobStatus.COMPLETED
            job.stage = "completed"
            job.chunk_count = len(chunks)
            self.repository.save_job(job)
        except Exception as exc:
            logger.exception("Ingestion failed for job %s", job_id)
            try:
                self.vector_store.delete_document(job.document_id)
            except Exception:
                logger.exception("Could not clean failed Qdrant ingestion %s", job_id)
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error = self._safe_error(exc)
            self.repository.save_job(job)

    def search(
        self, query: str, *, top_k: int, document_ids: list[str] | None
    ) -> SearchResponse:
        started = time.perf_counter()
        vector = self.embeddings.embed_query(query)
        hits = self.vector_store.search(
            vector, limit=top_k, document_ids=document_ids
        )
        return SearchResponse(
            query=query,
            hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            embedding_model_id=self.embeddings.model_id,
        )

    def chat(
        self,
        question: str,
        *,
        top_k: int | None,
        document_ids: list[str] | None,
        session_id: str | None,
    ) -> ChatResponse:
        started = time.perf_counter()
        requested_top_k = top_k or self.settings.chat_retrieval_top_k
        retrieval = self.search(
            question,
            top_k=requested_top_k,
            document_ids=document_ids,
        )
        retrieval_finished = time.perf_counter()
        if not retrieval.hits:
            total_ms = round((retrieval_finished - started) * 1000, 2)
            return ChatResponse(
                status="insufficient_evidence",
                answer="The indexed sources do not contain enough evidence to answer this question.",
                citations=[],
                latency_ms=total_ms,
                timings=ChatTimings(
                    retrieval_ms=retrieval.latency_ms,
                    generation_ms=0,
                    total_ms=total_ms,
                ),
                model_id=self.generator.model_id,
                embedding_model_id=self.embeddings.model_id,
                debug=ChatDebug(
                    requested_top_k=requested_top_k,
                    retrieved_chunks=[],
                    session_id=session_id,
                ),
            )

        generated = self.generator.generate(question, retrieval.hits)
        finished = time.perf_counter()
        by_evidence_id = {
            f"E{position}": hit
            for position, hit in enumerate(retrieval.hits, start=1)
        }
        citations: list[RagCitation] = []
        for evidence_id in generated.evidence_ids:
            hit = by_evidence_id[evidence_id]
            citations.append(
                RagCitation(
                    evidence_id=evidence_id,
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source_path=hit.filename,
                    title=hit.title,
                    page_numbers=hit.page_numbers,
                    heading_path=hit.heading_path,
                    score=hit.score,
                    excerpt=hit.text[:1_000],
                )
            )
        total_ms = round((finished - started) * 1000, 2)
        generation_ms = round((finished - retrieval_finished) * 1000, 2)
        return ChatResponse(
            status=generated.status,
            answer=generated.answer,
            citations=citations,
            usage=generated.usage,
            latency_ms=total_ms,
            timings=ChatTimings(
                retrieval_ms=retrieval.latency_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
            model_id=self.generator.model_id,
            embedding_model_id=self.embeddings.model_id,
            debug=ChatDebug(
                requested_top_k=requested_top_k,
                retrieved_chunks=retrieval.hits,
                cited_chunk_ids=[item.chunk_id for item in citations],
                generation_attempts=generated.attempts,
                generation_stop_reason=generated.stop_reason,
                session_id=session_id,
            ),
        )

    def _validate_upload(self, filename: str, content: bytes) -> str:
        if not content:
            raise UploadValidationError("The uploaded file is empty.")
        if len(content) > self.settings.max_upload_bytes:
            raise UploadValidationError(
                f"The upload exceeds the {self.settings.max_upload_bytes} byte limit."
            )
        actual_suffix = ("." + filename.rsplit(".", 1)[-1]).casefold() if "." in filename else ""
        if actual_suffix not in self.extractor.SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(self.extractor.SUPPORTED_SUFFIXES))
            raise UploadValidationError(f"Unsupported file type. Supported: {supported}.")
        return actual_suffix

    @staticmethod
    def _clean_title(title: str | None, filename: str) -> str:
        cleaned = (title or "").strip()
        return cleaned[:300] if cleaned else filename.rsplit(".", 1)[0][:300]

    def _read_upload_metadata(self, document_id: str) -> dict[str, str]:
        path = self.repository.artifact_dir(document_id) / "upload_metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, (DocumentExtractionError, UploadValidationError)):
            return str(exc)
        return f"{type(exc).__name__}: ingestion failed; see backend logs."
