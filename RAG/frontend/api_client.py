from __future__ import annotations

from typing import Any

import requests


class RagApiError(RuntimeError):
    pass


class RagApiClient:
    def __init__(self, base_url: str, *, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def upload(
        self,
        *,
        filename: str,
        content: bytes,
        media_type: str | None,
        title: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/documents",
            files={"file": (filename, content, media_type or "application/octet-stream")},
            data={"title": title or ""},
            timeout=max(self.timeout, 120),
        )

    def ingestion(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/ingestions/{job_id}")

    def documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")

    def search(
        self,
        *,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/search",
            json={
                "query": query,
                "top_k": top_k,
                "document_ids": document_ids or None,
            },
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise RagApiError(f"Cannot reach the RAG API at {self.base_url}.") from exc
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RagApiError(f"API {response.status_code}: {detail}")

