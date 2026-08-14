"""Shared, dependency-free helpers for SG-IA deployment tooling."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DOCUMENT_SUFFIXES = frozenset(
    {".pdf", ".docx", ".pptx", ".md", ".txt", ".csv", ".json"}
)


class DeploymentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    path: str
    wiki_path: str
    sha256: str
    title: str | None
    source: Path


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    path: Path
    source_root: Path
    documents: tuple[ManifestDocument, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str, *, field: str) -> str:
    normalized = value.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
    ):
        raise DeploymentError(f"Manifest {field} must be a safe relative path: {value!r}")
    return pure.as_posix()


def within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DeploymentError(f"Path escapes its configured root: {relative}") from exc
    return candidate


def load_manifest(path: Path) -> CorpusManifest:
    manifest_path = path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"Corpus manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise DeploymentError("Corpus manifest must use schema_version 1.")

    source_root_value = relative_path(
        str(payload.get("source_root", "material")), field="source_root"
    )
    source_root = within(ROOT, source_root_value)
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise DeploymentError("Corpus manifest must contain at least one document.")

    seen_paths: set[str] = set()
    seen_wiki_paths: set[str] = set()
    documents: list[ManifestDocument] = []
    for index, item in enumerate(raw_documents, start=1):
        if not isinstance(item, Mapping):
            raise DeploymentError(f"Manifest document {index} is not an object.")
        document_path = relative_path(str(item.get("path", "")), field="path")
        wiki_path = relative_path(
            str(item.get("wiki_path") or document_path), field="wiki_path"
        )
        source = within(source_root, document_path)
        if not source.is_file():
            raise DeploymentError(f"Manifest source does not exist: {source}")
        if source.suffix.casefold() not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise DeploymentError(f"Unsupported manifest document: {document_path}")
        if document_path.casefold() in seen_paths:
            raise DeploymentError(f"Duplicate manifest path: {document_path}")
        if wiki_path.casefold() in seen_wiki_paths:
            raise DeploymentError(f"Duplicate Wiki staging path: {wiki_path}")
        seen_paths.add(document_path.casefold())
        seen_wiki_paths.add(wiki_path.casefold())

        actual_hash = sha256_file(source)
        expected_hash = str(item.get("sha256", "")).strip().casefold()
        if expected_hash and expected_hash != actual_hash:
            raise DeploymentError(f"Hash mismatch for manifest source: {document_path}")
        title = item.get("title")
        documents.append(
            ManifestDocument(
                path=document_path,
                wiki_path=wiki_path,
                sha256=actual_hash,
                title=str(title).strip() if title not in (None, "") else None,
                source=source,
            )
        )

    return CorpusManifest(
        path=manifest_path,
        source_root=source_root,
        documents=tuple(documents),
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def request_json(
    method: str,
    url: str,
    *,
    payload: Any | None = None,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> Any:
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise DeploymentError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DeploymentError(f"Cannot reach {url}: {exc.reason}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"{url} returned an invalid JSON response.") from exc


def multipart_document(document: ManifestDocument) -> tuple[bytes, str]:
    boundary = f"sgia-{os.urandom(12).hex()}"
    newline = b"\r\n"
    media_type = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
    }.get(document.source.suffix.casefold(), "application/octet-stream")
    parts = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="title"\r\n\r\n',
        (document.title or "").encode("utf-8"),
        newline,
        f"--{boundary}\r\n".encode(),
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{document.source.name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {media_type}\r\n\r\n".encode(),
        document.source.read_bytes(),
        newline,
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), boundary
