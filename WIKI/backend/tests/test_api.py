from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.main import app, get_service


class FakeService:
    def __init__(self) -> None:
        self.updated_paths: object = "not-called"
        self.questions: list[str] = []
        self.repair_max_links: int | None = None

    def health(self):
        return {
            "status": "ok",
            "bedrock": {
                "configured": True,
                "model_id": "test-model",
                "region_name": "eu-test-1",
                "credentials_source": "test-only",
            },
        }

    def list_documents(self):
        return [
            {
                "relative_path": "example.md",
                "source_path": "raw/example.md",
                "status": "Pending",
                "size_bytes": 42,
                "modified_at": datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
            }
        ]

    def update_wiki(self, paths):
        self.updated_paths = paths
        selected = paths or ["example.md"]
        return {
            "processed": [
                {
                    "source_path": f"raw/{path.removeprefix('raw/')}",
                    "prompt": f"Ingest raw/{path.removeprefix('raw/')} into the wiki.",
                }
                for path in selected
            ],
            "skipped": [],
            "failed": [],
        }

    def ask(self, question):
        self.questions.append(question)
        return {
            "approach": "wiki",
            "status": "answered",
            "answer": "Grounded answer",
            "citations": [
                {"wiki_path": "sources/example.md", "source_paths": ["raw/example.md"]}
            ],
            "usage": {"inputTokens": 12, "outputTokens": 3},
            "latency_ms": 15.5,
            "model_id": "test-model",
            "confidence_score": 8.7,
            "debug": {
                "pages_read": ["sources/example.md"],
                "search_queries": ["example"],
                "guardrail": {
                    "applied": False,
                    "original_status": "answered",
                    "verification_available": True,
                    "reasons": [],
                },
            },
        }

    def list_wiki_pages(self):
        return [
            {
                "path": "sources/example.md",
                "title": "Example",
                "summary": "A summary",
                "source_paths": ["raw/example.md"],
            }
        ]

    def lint_wiki(self):
        return {
            "valid": True,
            "pages_checked": 1,
            "issues": [],
            "graph": {
                "pages": 1,
                "links": 0,
                "without_incoming": 1,
                "without_outgoing": 1,
                "isolated": 1,
            },
        }

    def repair_wiki_links(self, *, max_links):
        self.repair_max_links = max_links
        return {
            "links_added": [
                {
                    "source_path": "sources/example.md",
                    "target_path": "concepts/example.md",
                    "reason": "The concept is derived from this source summary.",
                }
            ],
            "pages_updated": ["concepts/example.md", "sources/example.md"],
            "graph_before": {
                "pages": 2,
                "links": 0,
                "without_incoming": 2,
                "without_outgoing": 2,
                "isolated": 2,
            },
            "graph_after": {
                "pages": 2,
                "links": 2,
                "without_incoming": 0,
                "without_outgoing": 0,
                "isolated": 0,
            },
            "usage": {"inputTokens": 20, "outputTokens": 5},
        }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeService()
        app.dependency_overrides[get_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_does_not_expose_credentials(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "bedrock_configured": True,
                "model_id": "test-model",
                "region": "eu-test-1",
            },
        )
        self.assertNotIn("credential", response.text.lower())
        self.assertNotIn("secret", response.text.lower())

    def test_list_documents_uses_raw_root_relative_paths(self) -> None:
        response = self.client.get("/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"][0]["relative_path"], "example.md")

    def test_update_without_body_processes_all_pending(self) -> None:
        response = self.client.post("/wiki/update")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.service.updated_paths)
        self.assertEqual(response.json()["processed"], ["raw/example.md"])

    def test_update_normalizes_and_deduplicates_paths(self) -> None:
        response = self.client.post(
            "/wiki/update", json={"paths": ["example.md", "example.md"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.updated_paths, ["example.md"])
        self.assertEqual(response.json()["processed"], ["raw/example.md"])

    def test_update_rejects_traversal(self) -> None:
        response = self.client.post("/wiki/update", json={"paths": ["../secret.txt"]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.service.updated_paths, "not-called")

    def test_chat_trims_question_and_returns_wiki_and_raw_citations(self) -> None:
        response = self.client.post("/chat", json={"question": "  What is this?  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["confidence_score"], 8.7)
        self.assertFalse(response.json()["debug"]["guardrail"]["applied"])
        self.assertEqual(self.service.questions, ["What is this?"])
        self.assertEqual(response.json()["answer"], "Grounded answer")
        payload = response.json()
        self.assertEqual(
            payload["citations"],
            [{"wiki_path": "sources/example.md", "source_paths": ["raw/example.md"]}],
        )
        self.assertEqual(payload["approach"], "wiki")
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["usage"]["inputTokens"], 12)
        self.assertEqual(payload["debug"]["pages_read"], ["sources/example.md"])

    def test_chat_rejects_blank_question(self) -> None:
        response = self.client.post("/chat", json={"question": "   "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.service.questions, [])

    def test_list_wiki_pages_returns_metadata(self) -> None:
        response = self.client.get("/wiki/pages")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pages"][0]["relative_path"], "sources/example.md")
        self.assertEqual(response.json()["pages"][0]["summary"], "A summary")
        self.assertEqual(response.json()["pages"][0]["source_paths"], ["raw/example.md"])

    def test_lint_endpoint_returns_deterministic_report(self) -> None:
        response = self.client.get("/wiki/lint")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "valid": True,
                "pages_checked": 1,
                "issues": [],
                "graph": {
                    "pages": 1,
                    "links": 0,
                    "without_incoming": 1,
                    "without_outgoing": 1,
                    "isolated": 1,
                },
            },
        )

    def test_semantic_link_repair_endpoint_returns_graph_improvement(self) -> None:
        response = self.client.post(
            "/wiki/lint/repair-links", json={"max_links": 6}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.repair_max_links, 6)
        payload = response.json()
        self.assertEqual(payload["graph_before"]["isolated"], 2)
        self.assertEqual(payload["graph_after"]["isolated"], 0)
        self.assertEqual(payload["links_added"][0]["target_path"], "concepts/example.md")

    def test_localhost_cors_preflight_is_allowed(self) -> None:
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], "http://localhost:8501"
        )

    def test_unexpected_service_error_is_sanitized(self) -> None:
        class BrokenService(FakeService):
            def ask(self, question):
                raise RuntimeError("secret-key-value")

        app.dependency_overrides[get_service] = BrokenService
        response = self.client.post("/chat", json={"question": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "Wiki service could not answer the question."}
        )
        self.assertNotIn("secret-key-value", response.text)


if __name__ == "__main__":
    unittest.main()
