from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.agent import WikiAgent
from backend.app.bedrock import ConverseTurn
from backend.app.config import BedrockSettings, load_settings
from backend.app.embeddings import EmbeddingError, EmbeddingResult, TitanEmbeddingClient
from backend.app.repository import WikiRepository
from backend.app.search import HybridWikiSearch


def wiki_page(title: str, source: str, body: str) -> str:
    return f"""---
title: {title}
page_type: source
updated: 2026-08-07
sources:
  - {source}
---
# {title}

{body}

## Sources

- {source}
"""


class SemanticEmbedder:
    model_id = "test-embedding-model"
    dimensions = 2
    max_input_characters = 45_000

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        folded = text.casefold()
        if "employee rest requirements" in folded or "video display" in folded:
            return EmbeddingResult((1.0, 0.0), 3)
        return EmbeddingResult((0.0, 1.0), 2)


class FailingEmbedder(SemanticEmbedder):
    def embed(self, text: str) -> EmbeddingResult:
        raise EmbeddingError("simulated embedding outage")


class ScriptedBedrock:
    def __init__(self, turns: list[ConverseTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("Unexpected Bedrock call")
        return self.turns.pop(0)


def assistant_turn(tool_id: str, name: str, inputs: dict[str, object]) -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": tool_id,
                        "name": name,
                        "input": inputs,
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 5, "outputTokens": 2},
        metrics={},
    )


class HybridSearchTests(unittest.TestCase):
    def make_repository(self, temp_dir: str) -> WikiRepository:
        backend_root = Path(temp_dir) / "backend"
        raw_root = backend_root / "raw"
        raw_root.mkdir(parents=True)
        (backend_root / "AGENTS.md").write_text("Maintain a grounded wiki.", encoding="utf-8")
        (raw_root / "safety.txt").write_text("Safety source", encoding="utf-8")
        (raw_root / "history.txt").write_text("History source", encoding="utf-8")
        repository = WikiRepository(backend_root)
        repository.write_wiki_pages(
            {
                "sources/safety.md": wiki_page(
                    "Office Screen Safety",
                    "raw/safety.txt",
                    "Workers using video display terminals must pause regularly.",
                ),
                "sources/history.md": wiki_page(
                    "Company History",
                    "raw/history.txt",
                    "The organization was founded many years ago.",
                ),
            }
        )
        return repository

    def test_similarity_finds_page_when_lexical_search_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = self.make_repository(temp_dir)
            self.assertEqual(
                repository.search_wiki("employee rest requirements"), []
            )
            embedder = SemanticEmbedder()
            searcher = HybridWikiSearch(repository, embedder)

            response = searcher.search("employee rest requirements", limit=2)

            self.assertEqual(response.mode, "hybrid")
            self.assertEqual(response.results[0].path, "sources/safety.md")
            self.assertGreater(response.embedding_input_tokens, 0)

    def test_content_hash_cache_reuses_and_invalidates_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = self.make_repository(temp_dir)
            embedder = SemanticEmbedder()
            cache_path = Path(temp_dir) / "semantic-cache.json"
            searcher = HybridWikiSearch(repository, embedder, cache_path=cache_path)

            searcher.search("employee rest requirements", limit=2)
            self.assertEqual(len(embedder.calls), 3)  # two pages plus query
            searcher.search("employee rest requirements", limit=2)
            self.assertEqual(len(embedder.calls), 4)  # query only

            repository.write_wiki_pages(
                {
                    "sources/safety.md": wiki_page(
                        "Office Screen Safety",
                        "raw/safety.txt",
                        "Revised video display guidance requires regular pauses.",
                    )
                }
            )
            searcher.search("employee rest requirements", limit=2)
            self.assertEqual(len(embedder.calls), 6)  # changed page plus query
            self.assertTrue(cache_path.is_file())

    def test_embedding_failure_falls_back_to_exact_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = self.make_repository(temp_dir)
            searcher = HybridWikiSearch(repository, FailingEmbedder())

            with self.assertLogs("backend.app.search", level="WARNING"):
                response = searcher.search("founded", limit=2)

            self.assertEqual(response.mode, "lexical_fallback")
            self.assertEqual(response.results[0].path, "sources/history.md")

    def test_agent_reports_hybrid_mode_and_embedding_usage(self) -> None:
        turns = [
            assistant_turn(
                "search",
                "search_wiki",
                {"query": "employee rest requirements", "limit": 2},
            ),
            assistant_turn("read", "read_wiki_page", {"path": "sources/safety.md"}),
            assistant_turn(
                "submit",
                "submit_answer",
                {
                    "status": "answered",
                    "answer": "Workers must take regular pauses.",
                    "citations": [
                        {
                            "wiki_path": "sources/safety.md",
                            "source_paths": ["raw/safety.txt"],
                        }
                    ],
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = self.make_repository(temp_dir)
            scripted = ScriptedBedrock(turns)
            searcher = HybridWikiSearch(repository, SemanticEmbedder())
            agent = WikiAgent(repository, scripted, searcher=searcher, max_steps=6)

            result = agent.answer("What rest requirements apply to employees?")

            self.assertEqual(result.search_modes, ("hybrid",))
            self.assertGreater(result.usage["embeddingInputTokens"], 0)
            modes = [
                block["toolResult"]["content"][0]["json"]["mode"]
                for call in scripted.calls
                for message in call["messages"]
                if message["role"] == "user"
                for block in message["content"]
                if "toolResult" in block
                and "mode" in block["toolResult"]["content"][0].get("json", {})
            ]
            self.assertTrue(modes)
            self.assertEqual(set(modes), {"hybrid"})

    def test_titan_client_normalizes_and_validates_response(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.request = None

            def invoke_model(self, **kwargs):
                self.request = kwargs
                vector = [3.0, 4.0] + [0.0] * 254
                return {
                    "body": io.BytesIO(
                        json.dumps(
                            {"embedding": vector, "inputTextTokenCount": 7}
                        ).encode("utf-8")
                    )
                }

        settings = BedrockSettings(
            project_root=Path.cwd(),
            region_name="eu-central-1",
            bedrock_model_id="answer-model",
            embedding_dimensions=256,
        )
        runtime = FakeRuntime()
        client = TitanEmbeddingClient(settings, client=runtime)

        result = client.embed("A semantic query")

        self.assertAlmostEqual(result.vector[0], 0.6)
        self.assertAlmostEqual(result.vector[1], 0.8)
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(runtime.request["modelId"], "amazon.titan-embed-text-v2:0")

    def test_embedding_configuration_is_generic_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "aws_credentials.json").write_text(
                json.dumps(
                    {
                        "region_name": "eu-central-1",
                        "bedrock_model_id": "answer-model",
                    }
                ),
                encoding="utf-8",
            )
            defaults = load_settings(root, environ={})
            disabled = load_settings(
                root,
                environ={
                    "LLM_WIKI_SEMANTIC_SEARCH_ENABLED": "false",
                    "BEDROCK_EMBEDDING_DIMENSIONS": "256",
                },
            )

            self.assertEqual(defaults.embedding_model_id, "amazon.titan-embed-text-v2:0")
            self.assertEqual(defaults.embedding_dimensions, 512)
            self.assertTrue(defaults.semantic_search_enabled)
            self.assertFalse(disabled.semantic_search_enabled)
            self.assertEqual(disabled.embedding_dimensions, 256)


if __name__ == "__main__":
    unittest.main()
