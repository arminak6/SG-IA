from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from .generation import AnswerGenerationError
from .models import (
    ChatRequest,
    ChatResponse,
    DocumentRecord,
    HealthResponse,
    IngestionJob,
    ModelConfigurationResponse,
    SearchRequest,
    SearchResponse,
    UploadAccepted,
)
from .runtime import get_service
from .service import UploadValidationError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SG-IA RAG API",
    version="1.2.0",
    description="API-first document ingestion, semantic retrieval, and grounded Q&A for SG-IA RAG.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8502", "http://127.0.0.1:8502"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    qdrant_ok = service.vector_store.health()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        pipeline_version=service.settings.pipeline_version,
        qdrant="reachable" if qdrant_ok else "unreachable",
        collection=service.settings.qdrant_collection,
        embedding_model_id=service.embeddings.model_id,
        generation_model_id=service.generator.model_id,
        confidence_model_id=(
            service.confidence_evaluator.model_id
            if service.confidence_evaluator is not None
            else None
        ),
    )


@app.get("/models", response_model=ModelConfigurationResponse)
def models() -> ModelConfigurationResponse:
    settings = get_service().settings
    return ModelConfigurationResponse(
        pipeline_version=settings.pipeline_version,
        extraction={
            "engine": settings.extraction_engine,
            "layout_model_id": settings.layout_model_id,
            "ocr_model_id": settings.ocr_model_id,
            "table_structure_model_id": settings.table_structure_model_id,
            "ocr_enabled": settings.docling_do_ocr,
        },
        embedding={
            "provider": "amazon-bedrock",
            "model_id": settings.embedding_model_id,
            "dimensions": settings.embedding_dimensions,
            "normalize": True,
        },
        chunking={
            "strategy": "structure-aware",
            "max_tokens": settings.chunk_max_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
        },
        retrieval={
            "strategy": "semantic-candidate-neighbor-rerank-coverage-v1.2",
            "candidate_pool_size": settings.chat_candidate_pool_size,
            "final_top_k": settings.chat_retrieval_top_k,
            "neighbor_window": settings.chat_neighbor_window,
            "coverage_retry_enabled": settings.chat_coverage_retry_enabled,
            "coverage_min_ratio": settings.chat_coverage_min_ratio,
            "max_attempts": settings.chat_max_retrieval_attempts,
        },
        generation={
            "provider": "amazon-bedrock",
            "api": "converse",
            "model_id": settings.generation_model_id,
            "temperature": settings.generation_temperature,
            "max_output_tokens": settings.generation_max_output_tokens,
        },
        verification={
            "provider": "amazon-bedrock",
            "api": "converse",
            "purpose": "rag-evidence-confidence",
            "enabled": settings.confidence_enabled,
            "model_id": settings.confidence_model_id,
            "temperature": 0.0,
            "max_output_tokens": settings.confidence_max_output_tokens,
            "max_evidence_characters": settings.confidence_max_evidence_characters,
        },
    )


@app.post(
    "/documents",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
) -> UploadAccepted:
    service = get_service()
    content = await file.read(service.settings.max_upload_bytes + 1)
    try:
        accepted = service.accept_upload(
            filename=file.filename or "document",
            content=content,
            title=title,
            media_type=file.content_type,
        )
    except UploadValidationError as exc:
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if len(content) > service.settings.max_upload_bytes
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    if accepted.schedule_ingestion:
        background_tasks.add_task(service.ingest, accepted.job.job_id)
    message = (
        "This content is already indexed; the existing document was reused."
        if accepted.duplicate
        else "Upload accepted and ingestion queued."
    )
    return UploadAccepted(
        job=accepted.job, duplicate=accepted.duplicate, message=message
    )


@app.get("/ingestions/{job_id}", response_model=IngestionJob)
def get_ingestion(job_id: str) -> IngestionJob:
    job = get_service().repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return get_service().repository.list_documents()


@app.get("/documents/{document_id}", response_model=DocumentRecord)
def get_document(document_id: str) -> DocumentRecord:
    document = get_service().repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return get_service().search(
            request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Retrieval failed ({type(exc).__name__}); see backend logs.",
        ) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return get_service().chat(
            request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
            session_id=request.session_id,
        )
    except AnswerGenerationError as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=503,
            detail=f"Answer generation is unavailable ({type(exc).__name__}); see backend logs.",
        ) from exc
    except Exception as exc:
        logger.exception("RAG question processing failed")
        raise HTTPException(
            status_code=502,
            detail=f"RAG question processing failed ({type(exc).__name__}); see backend logs.",
        ) from exc
