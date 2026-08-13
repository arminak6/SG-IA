from types import SimpleNamespace

from app import api
from app.models import (
    ChatDebug,
    ChatResponse,
    ChatTimings,
    IngestionJob,
    JobStatus,
)
from app.service import AcceptedUpload
from fastapi.testclient import TestClient


class FakeApiService:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            max_upload_bytes=1_000,
            qdrant_collection="test_collection",
        )
        self.embeddings = SimpleNamespace(model_id="fake-embeddings")
        self.generator = SimpleNamespace(model_id="fake-generation")
        self.settings.embedding_model_id = "fake-embeddings"
        self.settings.embedding_dimensions = 256
        self.settings.generation_model_id = "fake-generation"
        self.settings.generation_temperature = 0.1
        self.settings.generation_max_output_tokens = 500
        self.settings.extraction_engine = "docling"
        self.settings.layout_model_id = "layout"
        self.settings.ocr_model_id = "ocr"
        self.settings.table_structure_model_id = "table"
        self.settings.docling_do_ocr = False
        self.vector_store = SimpleNamespace(health=lambda: True)
        self.repository = SimpleNamespace(
            get_job=lambda job_id: self.job if job_id == self.job.job_id else None,
            list_documents=list,
            get_document=lambda document_id: None,
        )
        self.ingested: list[str] = []
        self.job = IngestionJob(
            job_id="job-1",
            document_id="doc-1",
            filename="guide.txt",
            source_hash="abc",
        )

    def accept_upload(self, **_kwargs) -> AcceptedUpload:
        return AcceptedUpload(
            job=self.job,
            duplicate=False,
            schedule_ingestion=True,
        )

    def ingest(self, job_id: str) -> None:
        self.ingested.append(job_id)
        self.job.status = JobStatus.COMPLETED
        self.job.stage = "completed"

    def chat(self, question: str, **_kwargs) -> ChatResponse:
        return ChatResponse(
            status="answered",
            answer=f"Grounded: {question}",
            citations=[],
            latency_ms=12.0,
            timings=ChatTimings(
                retrieval_ms=2.0, generation_ms=10.0, total_ms=12.0
            ),
            model_id="fake-generation",
            embedding_model_id="fake-embeddings",
            debug=ChatDebug(requested_top_k=8),
        )


def test_health_and_upload_contract(monkeypatch) -> None:
    service = FakeApiService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    client = TestClient(api.app)

    health = client.get("/health")
    upload = client.post(
        "/documents",
        files={"file": ("guide.txt", b"A useful procedure", "text/plain")},
        data={"title": "Guide"},
    )

    assert health.status_code == 200
    assert health.json()["qdrant"] == "reachable"
    assert health.json()["generation_model_id"] == "fake-generation"
    assert upload.status_code == 202
    assert upload.json()["job"]["document_id"] == "doc-1"
    assert service.ingested == ["job-1"]

    model_config = client.get("/models")
    assert model_config.status_code == 200
    assert model_config.json()["generation"]["model_id"] == "fake-generation"

    chat = client.post("/chat", json={"question": "What is the rule?"})
    assert chat.status_code == 200
    assert chat.json()["approach"] == "rag"
    assert chat.json()["answer"] == "Grounded: What is the rule?"


def test_missing_resources_return_404(monkeypatch) -> None:
    service = FakeApiService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    client = TestClient(api.app)

    assert client.get("/ingestions/unknown").status_code == 404
    assert client.get("/documents/unknown").status_code == 404
