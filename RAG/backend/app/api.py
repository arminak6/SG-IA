from __future__ import annotations

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

from .models import (
    DocumentRecord,
    HealthResponse,
    IngestionJob,
    SearchRequest,
    SearchResponse,
    UploadAccepted,
)
from .runtime import get_service
from .service import UploadValidationError

app = FastAPI(
    title="SG-IA RAG API",
    version="0.1.0",
    description="API-first document ingestion and vector retrieval for SG-IA RAG.",
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
        qdrant="reachable" if qdrant_ok else "unreachable",
        collection=service.settings.qdrant_collection,
        embedding_model_id=service.embeddings.model_id,
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
