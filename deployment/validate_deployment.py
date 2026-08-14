"""Validate unified SG-IA service readiness and corpus alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from common import ROOT, DeploymentError, load_manifest, request_json


def arguments() -> argparse.Namespace:
    environment = local_env()
    parser = argparse.ArgumentParser(description="Validate SG-IA deployment readiness.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "deployment/corpus-manifest.json")
    parser.add_argument(
        "--rag-url",
        default=f"http://127.0.0.1:{environment.get('RAG_API_PORT', '8001')}",
    )
    parser.add_argument(
        "--wiki-url",
        default=f"http://127.0.0.1:{environment.get('WIKI_API_PORT', '8002')}",
    )
    parser.add_argument(
        "--rag-ui-url",
        default=f"http://127.0.0.1:{environment.get('RAG_UI_PORT', '8502')}",
    )
    parser.add_argument(
        "--wiki-ui-url",
        default=f"http://127.0.0.1:{environment.get('WIKI_UI_PORT', '8503')}",
    )
    parser.add_argument(
        "--comparison-url",
        default=f"http://127.0.0.1:{environment.get('COMPARISON_UI_PORT', '8504')}",
    )
    parser.add_argument("--wait", type=float, default=0, help="Wait up to this many seconds for readiness.")
    parser.add_argument("--skip-manifest", action="store_true")
    return parser.parse_args()


def fetch_text(url: str, *, timeout: float = 10) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DeploymentError(f"Cannot reach {url}: {exc}") from exc


def wait_for_services(args: argparse.Namespace) -> None:
    endpoints = [
        f"{args.rag_url.rstrip('/')}/health",
        f"{args.wiki_url.rstrip('/')}/health",
        f"{args.rag_ui_url.rstrip('/')}/_stcore/health",
        f"{args.wiki_ui_url.rstrip('/')}/_stcore/health",
        f"{args.comparison_url.rstrip('/')}/_stcore/health",
    ]
    deadline = time.monotonic() + max(0, args.wait)
    while True:
        failures = []
        for endpoint in endpoints:
            try:
                fetch_text(endpoint, timeout=5)
            except DeploymentError as exc:
                failures.append(str(exc))
        if not failures:
            return
        if time.monotonic() >= deadline:
            raise DeploymentError("Services are not ready:\n- " + "\n- ".join(failures))
        time.sleep(2)


def local_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def validate_credentials() -> dict[str, Any]:
    environment = local_env()
    access_key = environment.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = environment.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if access_key and secret_key:
        if len(access_key) < 16 or len(secret_key) < 32:
            raise DeploymentError("AWS environment credentials appear to be placeholders.")
        return {"configured": True, "source": "environment"}
    configured_path = environment.get(
        "SGIA_AWS_CREDENTIALS_FILE", "./WIKI/aws_credentials.example.json"
    )
    credential_path = Path(configured_path)
    if not credential_path.is_absolute():
        credential_path = (ROOT / credential_path).resolve()
    try:
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            f"Configured AWS credential file is missing or unreadable: {credential_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DeploymentError("Configured AWS credential file is not a JSON object.")
    file_access_key = str(payload.get("aws_access_key_id", "")).strip()
    file_secret_key = str(payload.get("aws_secret_access_key", "")).strip()
    if len(file_access_key) < 16 or len(file_secret_key) < 32:
        raise DeploymentError(
            "AWS credentials are placeholders. Configure temporary credentials before Q&A or ingestion."
        )
    return {"configured": True, "source": "read_only_json"}


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    rag_documents = request_json("GET", f"{args.rag_url.rstrip('/')}/documents", timeout=30)
    wiki_payload = request_json("GET", f"{args.wiki_url.rstrip('/')}/documents", timeout=30)
    if not isinstance(rag_documents, list):
        raise DeploymentError("RAG returned an invalid document list.")
    if not isinstance(wiki_payload, Mapping) or not isinstance(wiki_payload.get("documents"), list):
        raise DeploymentError("WIKI returned an invalid document list.")
    rag_hashes = {
        str(item.get("source_hash", "")).casefold()
        for item in rag_documents
        if isinstance(item, Mapping)
    }
    wiki_status = {
        str(item.get("relative_path", "")).replace("\\", "/").casefold(): str(
            item.get("status", "")
        ).casefold()
        for item in wiki_payload["documents"]
        if isinstance(item, Mapping)
    }
    missing_rag = [item.path for item in manifest.documents if item.sha256 not in rag_hashes]
    missing_wiki = [
        item.wiki_path
        for item in manifest.documents
        if wiki_status.get(item.wiki_path.casefold()) not in {"ingested", "ready"}
    ]
    if missing_rag or missing_wiki:
        details = []
        if missing_rag:
            details.append(f"RAG missing {len(missing_rag)} manifest document(s)")
        if missing_wiki:
            details.append(f"WIKI missing {len(missing_wiki)} manifest document(s)")
        raise DeploymentError("Corpus alignment failed: " + "; ".join(details))
    return {
        "manifest_documents": len(manifest.documents),
        "rag_manifest_coverage": len(manifest.documents),
        "wiki_manifest_coverage": len(manifest.documents),
    }


def main() -> int:
    args = arguments()
    wait_for_services(args)
    rag_health = request_json("GET", f"{args.rag_url.rstrip('/')}/health", timeout=30)
    wiki_health = request_json("GET", f"{args.wiki_url.rstrip('/')}/health", timeout=30)
    if not isinstance(rag_health, Mapping) or rag_health.get("status") != "ok":
        raise DeploymentError("RAG health is not OK.")
    if not isinstance(wiki_health, Mapping) or wiki_health.get("status") != "ok":
        raise DeploymentError("WIKI health is not OK.")
    if not bool(wiki_health.get("bedrock_configured")):
        raise DeploymentError("WIKI reports that Bedrock is not configured.")
    wiki_model_id = str(wiki_health.get("model_id", "")).strip()
    if not wiki_model_id or "your-bedrock-model" in wiki_model_id.casefold():
        raise DeploymentError("WIKI reports a missing or placeholder Bedrock model ID.")
    lint = request_json("GET", f"{args.wiki_url.rstrip('/')}/wiki/lint", timeout=60)
    if not isinstance(lint, Mapping) or not bool(lint.get("valid")):
        raise DeploymentError("WIKI lint is not valid.")

    result: dict[str, Any] = {
        "status": "ready",
        "credentials": validate_credentials(),
        "rag": {
            "status": rag_health.get("status"),
            "qdrant": rag_health.get("qdrant"),
            "pipeline_version": rag_health.get("pipeline_version"),
        },
        "wiki": {
            "status": wiki_health.get("status"),
            "bedrock_configured": wiki_health.get("bedrock_configured"),
            "model_configured": True,
            "pages_checked": lint.get("pages_checked"),
        },
        "interfaces": {
            "rag": args.rag_ui_url,
            "wiki": args.wiki_ui_url,
            "comparison": args.comparison_url,
        },
    }
    if not args.skip_manifest:
        result["corpus"] = validate_manifest(args)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
