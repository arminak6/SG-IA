from __future__ import annotations

import threading
import time
import unittest
from typing import Any

import requests

from api_client import BackendAnswer, BackendClient, ask_both, check_both


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


class CoordinatedClient:
    def __init__(
        self,
        approach: str,
        barrier: threading.Barrier,
        *,
        fail: bool = False,
    ) -> None:
        self.approach = approach
        self.barrier = barrier
        self.fail = fail
        self.questions: list[tuple[str, str, int]] = []

    def chat(self, question: str, *, session_id: str, rag_top_k: int) -> BackendAnswer:
        self.questions.append((question, session_id, rag_top_k))
        self.barrier.wait(timeout=1)
        if self.fail:
            raise RuntimeError("simulated failure")
        return BackendAnswer(
            approach=self.approach,
            status="answered",
            answer=f"{self.approach}: {question}",
        )


class ApiClientTests(unittest.TestCase):
    def test_normalizes_rag_response_and_sends_supported_top_k(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "approach": "rag",
                        "status": "answered",
                        "answer": "Grounded RAG answer",
                        "citations": [
                            {
                                "evidence_id": "E1",
                                "source_path": "guide.pdf",
                                "page_numbers": [3],
                            }
                        ],
                        "usage": {"input_tokens": 10},
                        "latency_ms": 120.5,
                        "timings": {"retrieval_ms": 20.0, "total_ms": 120.5},
                        "model_id": "generation-model",
                        "embedding_model_id": "embedding-model",
                        "confidence_score": None,
                        "debug": {"retrieval_attempts": 1},
                    }
                )
            ]
        )
        client = BackendClient("rag", "http://rag", session=session)

        result = client.chat("  What is the rule?  ", session_id="session-1", rag_top_k=9)

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "Grounded RAG answer")
        self.assertEqual(result.citations[0]["evidence_id"], "E1")
        self.assertEqual(result.server_latency_ms, 120.5)
        self.assertEqual(
            session.calls[0]["json"],
            {"question": "What is the rule?", "session_id": "session-1", "top_k": 9},
        )

    def test_normalizes_wiki_response_without_rag_only_fields(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "approach": "wiki",
                        "status": "answered",
                        "answer": "Grounded Wiki answer",
                        "citations": [
                            {"wiki_path": "policies/rule.md", "source_paths": ["guide.pdf"]}
                        ],
                        "latency_ms": 88.0,
                        "confidence_score": 8.5,
                        "debug": {"pages_read": ["policies/rule.md"]},
                    }
                )
            ]
        )
        client = BackendClient("wiki", "http://wiki/", session=session)

        result = client.chat("Question", session_id="session-1")

        self.assertEqual(result.approach, "wiki")
        self.assertEqual(result.confidence_score, 8.5)
        self.assertEqual(result.citations[0]["wiki_path"], "policies/rule.md")
        self.assertNotIn("top_k", session.calls[0]["json"])

    def test_ask_both_is_concurrent_and_preserves_partial_success(self) -> None:
        barrier = threading.Barrier(2)
        wiki = CoordinatedClient("wiki", barrier)
        rag = CoordinatedClient("rag", barrier, fail=True)

        results = ask_both(
            "Same question",
            session_id="comparison-1",
            rag_top_k=10,
            clients={"wiki": wiki, "rag": rag},  # type: ignore[arg-type]
        )

        self.assertEqual(results["wiki"].answer, "wiki: Same question")
        self.assertTrue(results["wiki"].ok)
        self.assertFalse(results["rag"].ok)
        self.assertIn("RuntimeError", results["rag"].error or "")
        self.assertEqual(wiki.questions, [("Same question", "comparison-1", 10)])
        self.assertEqual(rag.questions, [("Same question", "comparison-1", 10)])

    def test_health_checks_both_backends(self) -> None:
        wiki_session = FakeSession([FakeResponse({"status": "ok", "model_id": "wiki"})])
        rag_session = FakeSession([FakeResponse({"status": "ok", "qdrant": "reachable"})])
        health = check_both(
            clients={
                "wiki": BackendClient("wiki", "http://wiki", session=wiki_session),
                "rag": BackendClient("rag", "http://rag", session=rag_session),
            }
        )

        self.assertTrue(health["wiki"].healthy)
        self.assertTrue(health["rag"].healthy)

    def test_http_error_keeps_backend_detail(self) -> None:
        session = FakeSession([FakeResponse({"detail": "generation unavailable"}, 503)])
        client = BackendClient("rag", "http://rag", session=session)

        started = time.perf_counter()
        results = ask_both(
            "Question",
            session_id="comparison-1",
            clients={
                "wiki": CoordinatedClient("wiki", threading.Barrier(1)),  # type: ignore[dict-item]
                "rag": client,
            },
        )

        self.assertLess(time.perf_counter() - started, 1)
        self.assertIn("HTTP 503", results["rag"].error or "")
        self.assertIn("generation unavailable", results["rag"].error or "")


if __name__ == "__main__":
    unittest.main()
