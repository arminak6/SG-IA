"""Build a local shared-corpus manifest by matching staged Wiki sources by hash."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import (
    ROOT,
    SUPPORTED_DOCUMENT_SUFFIXES,
    DeploymentError,
    sha256_file,
    write_json_atomic,
)


EXCLUDED_WIKI_DIRECTORIES = frozenset({"manager-actions", "manager-knowledge"})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match WIKI/backend/raw files to material/ by SHA-256 and write the "
            "ignored deployment/corpus-manifest.json file."
        )
    )
    parser.add_argument("--source-root", type=Path, default=ROOT / "material")
    parser.add_argument("--wiki-raw-root", type=Path, default=ROOT / "WIKI/backend/raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deployment/corpus-manifest.json"
    )
    return parser.parse_args()


def document_files(root: Path, *, exclude_manager: bool = False) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if exclude_manager and relative.parts and relative.parts[0].casefold() in EXCLUDED_WIKI_DIRECTORIES:
            continue
        result.append(path)
    return sorted(result, key=lambda item: item.relative_to(root).as_posix().casefold())


def main() -> int:
    args = arguments()
    source_root = args.source_root.expanduser().resolve()
    wiki_root = args.wiki_raw_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_root.is_dir() or not wiki_root.is_dir():
        raise DeploymentError("Both source and Wiki raw roots must exist.")

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for source in document_files(source_root):
        by_hash[sha256_file(source)].append(source)

    documents: list[dict[str, str | None]] = []
    problems: list[str] = []
    for wiki_source in document_files(wiki_root, exclude_manager=True):
        digest = sha256_file(wiki_source)
        matches = by_hash.get(digest, [])
        if len(matches) > 1:
            same_name = [item for item in matches if item.name.casefold() == wiki_source.name.casefold()]
            if len(same_name) == 1:
                matches = same_name
        if len(matches) != 1:
            problems.append(
                f"{wiki_source.relative_to(wiki_root).as_posix()}: "
                f"expected one material match, found {len(matches)}"
            )
            continue
        source = matches[0]
        documents.append(
            {
                "path": source.relative_to(source_root).as_posix(),
                "wiki_path": wiki_source.relative_to(wiki_root).as_posix(),
                "sha256": digest,
                "title": None,
            }
        )

    if problems:
        raise DeploymentError("Cannot create an exact manifest:\n- " + "\n- ".join(problems))
    if not documents:
        raise DeploymentError("No staged Wiki documents matched the shared material corpus.")

    source_relative = source_root.relative_to(ROOT).as_posix()
    write_json_atomic(
        output,
        {
            "schema_version": 1,
            "source_root": source_relative,
            "documents": sorted(documents, key=lambda item: str(item["path"]).casefold()),
        },
    )
    print(f"Created {output} with {len(documents)} matched documents.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
