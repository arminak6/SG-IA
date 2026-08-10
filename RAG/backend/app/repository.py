from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from pydantic import BaseModel

from .config import Settings
from .models import DocumentRecord, IngestionJob, utc_now

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


class LocalRepository:
    """Small, inspectable metadata store for the single-worker first milestone."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()
        self._lock = threading.RLock()

    @staticmethod
    def safe_filename(filename: str) -> str:
        name = Path(filename or "document").name.strip() or "document"
        return _SAFE_FILENAME.sub("_", name)[:180]

    def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        target_dir = self.settings.uploads_root / document_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / self.safe_filename(filename)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        return target

    def upload_path(self, document_id: str) -> Path:
        directory = self.settings.uploads_root / document_id
        matches = [path for path in directory.iterdir() if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one upload for document {document_id}")
        return matches[0]

    def artifact_dir(self, document_id: str) -> Path:
        path = self.settings.artifacts_root / document_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_job(self, job: IngestionJob) -> None:
        with self._lock:
            job.updated_at = utc_now()
            self._write_model(self.settings.jobs_root / f"{job.job_id}.json", job)

    def get_job(self, job_id: str) -> IngestionJob | None:
        path = self.settings.jobs_root / f"{job_id}.json"
        if not path.exists():
            return None
        return IngestionJob.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[IngestionJob]:
        jobs = [
            IngestionJob.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.settings.jobs_root.glob("*.json")
        ]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def find_active_job_by_hash(self, source_hash: str) -> IngestionJob | None:
        return next(
            (
                job
                for job in self.list_jobs()
                if job.source_hash == source_hash
                and job.status.value in {"queued", "processing"}
            ),
            None,
        )

    def save_document(self, document: DocumentRecord) -> None:
        with self._lock:
            self._write_model(
                self.settings.documents_root / f"{document.document_id}.json", document
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        path = self.settings.documents_root / f"{document_id}.json"
        if not path.exists():
            return None
        return DocumentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_documents(self) -> list[DocumentRecord]:
        records = [
            DocumentRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.settings.documents_root.glob("*.json")
        ]
        return sorted(records, key=lambda item: item.indexed_at, reverse=True)

    def find_document_by_hash(self, source_hash: str) -> DocumentRecord | None:
        return next(
            (
                document
                for document in self.list_documents()
                if document.source_hash == source_hash
            ),
            None,
        )

    def save_artifact_model(
        self, document_id: str, filename: str, model: BaseModel
    ) -> Path:
        path = self.artifact_dir(document_id) / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    @staticmethod
    def _write_model(path: Path, model: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
