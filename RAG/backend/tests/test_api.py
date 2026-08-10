from types import SimpleNamespace

from app import api
from app.models import IngestionJob, JobStatus
from app.service import AcceptedUpload
from fastapi.testclient import TestClient


class FakeApiService:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            max_upload_bytes=1_000,
            qdrant_collection="test_collection",
        )
        self.embeddings = SimpleNamespace(model_id="fake-embeddings")
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
    assert upload.status_code == 202
    assert upload.json()["job"]["document_id"] == "doc-1"
    assert service.ingested == ["job-1"]


def test_missing_resources_return_404(monkeypatch) -> None:
    service = FakeApiService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    client = TestClient(api.app)

    assert client.get("/ingestions/unknown").status_code == 404
    assert client.get("/documents/unknown").status_code == 404

