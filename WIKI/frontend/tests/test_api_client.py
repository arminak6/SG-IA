import unittest
from unittest.mock import Mock

import requests

from frontend.api_client import WikiApiClient, WikiApiError, WikiApiUnavailable


def response(payload, *, status_code=200):
    result = Mock()
    result.ok = 200 <= status_code < 400
    result.status_code = status_code
    result.json.return_value = payload
    result.text = ""
    return result


class WikiApiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Mock()
        self.client = WikiApiClient("http://api.example/", session=self.session)

    def test_list_documents_parses_envelope(self) -> None:
        self.session.request.return_value = response(
            {
                "documents": [
                    {
                        "relative_path": "research/article.md",
                        "status": "ingested",
                        "size_bytes": 42,
                        "modified_at": "2026-07-22T10:00:00Z",
                    }
                ]
            }
        )

        documents = self.client.list_documents()

        self.assertEqual(documents[0].relative_path, "research/article.md")
        self.assertTrue(documents[0].is_ingested)
        self.assertEqual(documents[0].size_bytes, 42)

    def test_update_sends_selected_paths_and_parses_per_file_results(self) -> None:
        self.session.request.return_value = response(
            {
                "processed": [
                    {"source_path": "new.md", "message": "Created one page"}
                ],
                "skipped": [{"source_path": "current.md", "reason": "Current"}],
                "failed": [{"source_path": "bad.md", "error": "Unsupported"}],
            }
        )

        result = self.client.update_wiki(["new.md", "bad.md"])

        _, _, kwargs = self.session.request.mock_calls[0]
        self.assertEqual(kwargs["json"], {"paths": ["new.md", "bad.md"]})
        self.assertEqual(result.processed[0].path, "new.md")
        self.assertEqual(result.skipped[0].reason, "Current")
        self.assertEqual(result.failed[0].error, "Unsupported")

    def test_chat_parses_answer_and_citations(self) -> None:
        self.session.request.return_value = response(
            {
                "answer": "The answer.",
                "citations": [
                    {"wiki_path": "wiki/topic.md", "source_paths": ["raw/a.txt"]}
                ],
                "confidence_score": 8.7,
            }
        )

        result = self.client.chat("What is it?", session_id="session-123")

        self.assertEqual(result.answer, "The answer.")
        self.assertEqual(
            result.citations,
            ("wiki/topic.md (sources: raw/a.txt)",),
        )
        self.assertEqual(result.confidence_score, 8.7)
        _, _, kwargs = self.session.request.mock_calls[0]
        self.assertEqual(
            kwargs["json"],
            {"question": "What is it?", "session_id": "session-123"},
        )
        self.assertEqual(kwargs["timeout"], (3.05, 600.0))

    def test_chat_rejects_out_of_range_confidence(self) -> None:
        self.session.request.return_value = response(
            {"answer": "The answer.", "citations": [], "confidence_score": 11}
        )

        with self.assertRaises(WikiApiError):
            self.client.chat("What is it?")

    def test_connection_failure_has_a_clear_error(self) -> None:
        self.session.request.side_effect = requests.ConnectionError("offline")

        with self.assertRaises(WikiApiUnavailable) as context:
            self.client.health()

        self.assertIn("http://api.example", str(context.exception))

    def test_health_allows_backend_status_scan_to_finish(self) -> None:
        self.session.request.return_value = response({"status": "ok"})

        self.client.health()

        _, _, kwargs = self.session.request.mock_calls[0]
        self.assertEqual(kwargs["timeout"], (1.0, 15.0))

    def test_http_error_uses_fastapi_detail(self) -> None:
        self.session.request.return_value = response(
            {"detail": "No wiki pages are available."}, status_code=409
        )

        with self.assertRaises(WikiApiError) as context:
            self.client.chat("Question")

        self.assertIn("No wiki pages are available", str(context.exception))


if __name__ == "__main__":
    unittest.main()
