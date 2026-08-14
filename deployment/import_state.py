"""Safely restore an SG-IA export into an empty deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from common import ROOT, DeploymentError
from export_state import VOLUME_ARCHIVES, WIKI_STATE_PATHS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a verified SG-IA state export into empty Docker volumes and "
            "empty local knowledge directories. Existing state is never deleted."
        )
    )
    parser.add_argument("backup", type=Path)
    parser.add_argument(
        "--confirm-empty-restore",
        action="store_true",
        help="Required explicit acknowledgement that only empty targets may be populated.",
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
            "Stop all SG-IA containers before restore. "
            f"Still running: {', '.join(names)}"
        )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_metadata(backup: Path) -> Mapping[str, Any]:
    metadata_path = backup / "state-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Backup metadata is missing or unreadable.") from exc
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != 1:
        raise DeploymentError("Backup metadata schema is unsupported.")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise DeploymentError("Backup metadata has no file inventory.")
    for filename, values in files.items():
        if not isinstance(filename, str) or not isinstance(values, Mapping):
            raise DeploymentError("Backup file inventory is invalid.")
        path = backup / filename
        if not path.is_file() or file_hash(path) != values.get("sha256"):
            raise DeploymentError(f"Backup file failed SHA-256 verification: {filename}")
    return metadata


def volume_state(volume: str) -> str:
    inspected = docker("volume", "inspect", volume, check=False)
    if inspected.returncode != 0:
        return "missing"
    result = docker(
        "run",
        "--rm",
        "-v",
        f"{volume}:/target",
        "python:3.11-slim",
        "sh",
        "-c",
        'test -z "$(find /target -mindepth 1 -maxdepth 1 -print -quit)"',
        check=False,
    )
    return "empty" if result.returncode == 0 else "nonempty"


def meaningful_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.name != ".gitkeep"
    ]


def validate_archive_members(archive: Path, destination: Path) -> list[tarfile.TarInfo]:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise DeploymentError(f"Unsafe archive path: {member.name}") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise DeploymentError(f"Unsupported archive member: {member.name}")
        return members


def safe_extract(archive: Path, destination: Path) -> None:
    members = validate_archive_members(archive, destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, members=members)


def validate_local_targets() -> None:
    for relative in WIKI_STATE_PATHS:
        files = meaningful_files(ROOT / relative)
        if files:
            raise DeploymentError(
                f"Restore target is not empty: {relative} ({len(files)} file(s))."
            )


def validate_material_merge(extracted_root: Path) -> None:
    extracted_material = extracted_root / "material"
    if not extracted_material.exists():
        return
    for source in extracted_material.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(extracted_material)
        destination = ROOT / "material" / relative
        if destination.is_file() and file_hash(destination) != file_hash(source):
            raise DeploymentError(
                f"Material restore would overwrite different content: {relative.as_posix()}"
            )


def restore_volume(volume: str, archive: Path) -> None:
    docker(
        "run",
        "--rm",
        "-v",
        f"{volume}:/target",
        "-v",
        f"{archive.parent.resolve()}:/backup:ro",
        "python:3.11-slim",
        "tar",
        "-xzf",
        f"/backup/{archive.name}",
        "-C",
        "/target",
    )


def copy_tree_without_overwrite(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)


def main() -> int:
    args = arguments()
    if not args.confirm_empty_restore:
        raise DeploymentError("Pass --confirm-empty-restore to authorize populating empty targets.")
    ensure_stopped()
    backup = args.backup.expanduser().resolve()
    metadata = verified_metadata(backup)
    if metadata.get("credentials_included") is not False:
        raise DeploymentError("Backup does not explicitly confirm credential exclusion.")

    manifest_target = ROOT / "deployment/corpus-manifest.json"
    manifest_source = backup / "corpus-manifest.json"
    if manifest_target.exists() and file_hash(manifest_target) != file_hash(manifest_source):
        raise DeploymentError("Local corpus manifest differs; refusing to overwrite it.")
    validate_local_targets()

    with tempfile.TemporaryDirectory(prefix="sgia-restore-") as temporary:
        temporary_root = Path(temporary)
        safe_extract(backup / "wiki-state.tar.gz", temporary_root / "wiki")
        safe_extract(backup / "manifest-corpus.tar.gz", temporary_root / "corpus")
        validate_material_merge(temporary_root / "corpus")
        for filename in VOLUME_ARCHIVES.values():
            validate_archive_members(backup / filename, Path("/target"))

        # Finish every read-only archive and local-path check before creating or
        # populating Docker volumes. A rejected restore must leave no partial state.
        states = {volume: volume_state(volume) for volume in VOLUME_ARCHIVES}
        for volume, state in states.items():
            if state == "nonempty":
                raise DeploymentError(
                    f"Docker volume is not empty; refusing to overwrite it: {volume}"
                )
        for volume, state in states.items():
            if state == "missing":
                docker("volume", "create", volume)

        for volume, filename in VOLUME_ARCHIVES.items():
            restore_volume(volume, backup / filename)

        for relative in WIKI_STATE_PATHS:
            copy_tree_without_overwrite(
                temporary_root / "wiki" / relative,
                ROOT / relative,
            )
        copy_tree_without_overwrite(
            temporary_root / "corpus" / "material",
            ROOT / "material",
        )

    if not manifest_target.exists():
        shutil.copy2(manifest_source, manifest_target)

    print("Restored SG-IA state into previously empty targets.")
    print("Configure fresh credentials, then start and validate the unified stack.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
