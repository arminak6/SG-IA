"""Ingest one manifest-defined corpus into both RAG and WIKI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from common import (
    ROOT,
    CorpusManifest,
    DeploymentError,
    load_manifest,
    multipart_document,
    request_json,
    sha256_file,
    within,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage and ingest the same manifest documents into RAG and WIKI."
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "deployment/corpus-manifest.json"
    )
    parser.add_argument("--rag-url", default="http://127.0.0.1:8001")
    parser.add_argument("--wiki-url", default="http://127.0.0.1:8002")
    parser.add_argument("--wiki-raw-root", type=Path, default=ROOT / "WIKI/backend/raw")
    parser.add_argument("--job-timeout", type=float, default=1800)
    parser.add_argument("--api-timeout", type=float, default=3600)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--skip-wiki", action="store_true")
    parser.add_argument(
        "--replace-staged-wiki-source",
        action="store_true",
        help="Explicitly replace a different file already staged at a manifest Wiki path.",
    )
    return parser.parse_args()


def stage_wiki_sources(
    manifest: CorpusManifest,
    wiki_raw_root: Path,
    *,
    replace: bool,
) -> list[str]:
    wiki_root = wiki_raw_root.expanduser().resolve()
    wiki_root.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for document in manifest.documents:
        destination = within(wiki_root, document.wiki_path)
        if destination.is_file():
            if sha256_file(destination) == document.sha256:
                staged.append(document.wiki_path)
                continue
            if not replace:
                raise DeploymentError(
                    f"Refusing to overwrite different staged Wiki source: {document.wiki_path}. "
                    "Use --replace-staged-wiki-source only after reviewing the manifest."
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(document.source, destination)
        if sha256_file(destination) != document.sha256:
            raise DeploymentError(f"Staged Wiki source failed hash verification: {document.wiki_path}")
        staged.append(document.wiki_path)
    return staged


def wait_for_rag_job(rag_url: str, job_id: str, *, timeout: float) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request_json(
            "GET", f"{rag_url.rstrip('/')}/ingestions/{job_id}", timeout=30
        )
        if not isinstance(job, Mapping):
            raise DeploymentError(f"RAG returned an invalid job response for {job_id}.")
        status = str(job.get("status", "")).casefold()
        if status == "completed":
            return job
        if status == "failed":
            raise DeploymentError(
                f"RAG ingestion failed for {job.get('filename', job_id)}: "
                f"{job.get('error') or 'unknown error'}"
            )
        time.sleep(2)
    raise DeploymentError(f"Timed out waiting for RAG ingestion job {job_id}.")


def ingest_rag(manifest: CorpusManifest, rag_url: str, *, timeout: float) -> dict[str, int]:
    base_url = rag_url.rstrip("/")
    existing = request_json("GET", f"{base_url}/documents", timeout=30)
    if not isinstance(existing, list):
        raise DeploymentError("RAG returned an invalid document list.")
    indexed_hashes = {
        str(item.get("source_hash", "")).casefold()
        for item in existing
        if isinstance(item, Mapping)
    }
    summary = {"processed": 0, "skipped": 0}
    for position, document in enumerate(manifest.documents, start=1):
        if document.sha256 in indexed_hashes:
            summary["skipped"] += 1
            continue
        body, boundary = multipart_document(document)
        accepted = request_json(
            "POST",
            f"{base_url}/documents",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=max(120, timeout),
        )
        if not isinstance(accepted, Mapping) or not isinstance(accepted.get("job"), Mapping):
            raise DeploymentError(f"RAG returned an invalid upload response for {document.path}.")
        job = accepted["job"]
        job_id = str(job.get("job_id", ""))
        if not job_id:
            raise DeploymentError(f"RAG returned no job ID for {document.path}.")
        wait_for_rag_job(base_url, job_id, timeout=timeout)
        indexed_hashes.add(document.sha256)
        summary["processed"] += 1
        print(f"RAG [{position}/{len(manifest.documents)}] indexed.")
    return summary


def ingest_wiki(
    manifest: CorpusManifest,
    wiki_url: str,
    wiki_raw_root: Path,
    *,
    replace: bool,
    timeout: float,
) -> dict[str, int]:
    staged = stage_wiki_sources(manifest, wiki_raw_root, replace=replace)
    result = request_json(
        "POST",
        f"{wiki_url.rstrip('/')}/wiki/update",
        payload={"paths": staged},
        timeout=timeout,
    )
    if not isinstance(result, Mapping):
        raise DeploymentError("WIKI returned an invalid update response.")
    failed = result.get("failed", [])
    if isinstance(failed, list) and failed:
        safe_failures = [
            {
                "path": str(item.get("path", "unknown")),
                "error": str(item.get("error", "unknown error")),
            }
            for item in failed
            if isinstance(item, Mapping)
        ]
        raise DeploymentError(
            "WIKI ingestion failures: " + json.dumps(safe_failures, ensure_ascii=True)
        )
    processed = result.get("processed", [])
    skipped = result.get("skipped", [])
    return {
        "processed": len(processed) if isinstance(processed, list) else 0,
        "skipped": len(skipped) if isinstance(skipped, list) else 0,
    }


def main() -> int:
    args = arguments()
    manifest = load_manifest(args.manifest)
    if args.skip_rag and args.skip_wiki:
        raise DeploymentError("Cannot skip both RAG and WIKI.")
    summary: dict[str, Any] = {"manifest_documents": len(manifest.documents)}
    if not args.skip_rag:
        request_json("GET", f"{args.rag_url.rstrip('/')}/health", timeout=30)
        summary["rag"] = ingest_rag(
            manifest,
            args.rag_url,
            timeout=args.job_timeout,
        )
    if not args.skip_wiki:
        request_json("GET", f"{args.wiki_url.rstrip('/')}/health", timeout=30)
        summary["wiki"] = ingest_wiki(
            manifest,
            args.wiki_url,
            args.wiki_raw_root,
            replace=args.replace_staged_wiki_source,
            timeout=args.api_timeout,
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
