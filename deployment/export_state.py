"""Export portable SG-IA knowledge state without credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, DeploymentError, load_manifest, write_json_atomic


VOLUME_ARCHIVES = {
    "sgia_rag_qdrant_data": "rag-qdrant-data.tar.gz",
    "sgia_rag_data": "rag-application-data.tar.gz",
}
WIKI_STATE_PATHS = (
    Path("WIKI/backend/raw"),
    Path("WIKI/backend/wiki"),
    Path("WIKI/backend/feedback"),
)


def arguments() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Export RAG Docker volumes, WIKI state, and manifest-selected corpus. "
            "All SG-IA containers must be stopped for a consistent snapshot."
        )
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deployment/backups" / timestamp
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "deployment/corpus-manifest.json"
    )
    return parser.parse_args()


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args],
            check=check,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise DeploymentError("Docker is not installed or is not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Docker command failed").strip()
        raise DeploymentError(detail) from exc


def ensure_stopped() -> None:
    running = docker("ps", "--filter", "name=sgia-", "--format", "{{.Names}}").stdout
    names = [line.strip() for line in running.splitlines() if line.strip()]
    if names:
        raise DeploymentError(
            "Stop all SG-IA containers before export for a consistent snapshot. "
            f"Still running: {', '.join(names)}"
        )


def export_volume(volume: str, archive: Path) -> None:
    docker("volume", "inspect", volume)
    docker(
        "run",
        "--rm",
        "-v",
        f"{volume}:/source:ro",
        "-v",
        f"{archive.parent.resolve()}:/backup",
        "python:3.11-slim",
        "tar",
        "-czf",
        f"/backup/{archive.name}",
        "-C",
        "/source",
        ".",
    )
    if not archive.is_file() or archive.stat().st_size == 0:
        raise DeploymentError(f"Docker volume export was not created: {archive}")


def archive_wiki_state(archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for relative in WIKI_STATE_PATHS:
            source = ROOT / relative
            if source.exists():
                handle.add(source, arcname=relative.as_posix(), recursive=True)


def archive_corpus(archive: Path, manifest_path: Path) -> int:
    manifest = load_manifest(manifest_path)
    with tarfile.open(archive, "w:gz") as handle:
        for document in manifest.documents:
            arcname = Path("material") / document.path
            handle.add(document.source, arcname=arcname.as_posix(), recursive=False)
    return len(manifest.documents)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    ensure_stopped()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise DeploymentError(f"Backup output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)

    archives: list[Path] = []
    for volume, filename in VOLUME_ARCHIVES.items():
        archive = output / filename
        export_volume(volume, archive)
        archives.append(archive)

    wiki_archive = output / "wiki-state.tar.gz"
    archive_wiki_state(wiki_archive)
    archives.append(wiki_archive)

    corpus_archive = output / "manifest-corpus.tar.gz"
    corpus_count = archive_corpus(corpus_archive, args.manifest)
    archives.append(corpus_archive)

    manifest_copy = output / "corpus-manifest.json"
    shutil.copy2(args.manifest.expanduser().resolve(), manifest_copy)
    archives.append(manifest_copy)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "credentials_included": False,
        "manifest_documents": corpus_count,
        "docker_volumes": VOLUME_ARCHIVES,
        "files": {
            path.name: {"sha256": file_hash(path), "size_bytes": path.stat().st_size}
            for path in archives
        },
    }
    write_json_atomic(output / "state-metadata.json", metadata)
    print(f"Exported portable SG-IA state to {output}")
    print("Credentials were intentionally excluded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
