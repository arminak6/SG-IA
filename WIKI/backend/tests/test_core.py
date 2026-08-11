from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.agent import (
    AgentValidationError,
    AnswerResult,
    Citation,
    IngestionResult,
    WikiAgent,
)
from backend.app.bedrock import BedrockConverseClient, BedrockError, ConverseTurn
from backend.app.config import BedrockSettings, load_settings
from backend.app.confidence import ConfidenceEvaluation
from backend.app.repository import RepositoryError, UnsafePathError, WikiRepository
from backend.app.service import WikiService


class ScriptedBedrock:
    def __init__(self, turns: list[ConverseTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("Unexpected Bedrock call")
        return self.turns.pop(0)


class StaticConfidenceEvaluator:
    def evaluate(self, question, result):
        return ConfidenceEvaluation(
            score=8.6,
            usage={"inputTokens": 2, "outputTokens": 1},
            claim_support=1.0,
            question_coverage=1.0,
            source_consistency=1.0,
            evidence_quality=1.0,
            abstention_score=1.0,
            has_unsupported_material_claim=False,
            has_unexplained_conflict=False,
            response_language="english",
        )


class BrokenConfidenceEvaluator:
    def evaluate(self, question, result):
        raise RuntimeError("verification unavailable")


class RejectingConfidenceEvaluator:
    def __init__(self, *, language="english"):
        self.language = language

    def evaluate(self, question, result):
        return ConfidenceEvaluation(
            score=9.3 if result.status == "insufficient_knowledge" else 4.0,
            usage={"inputTokens": 2, "outputTokens": 1},
            claim_support=0.4,
            question_coverage=0.9,
            source_consistency=1.0,
            evidence_quality=0.8,
            abstention_score=9.3,
            has_unsupported_material_claim=True,
            has_unexplained_conflict=False,
            response_language=self.language,
        )


def assistant_turn(*content: dict[str, object]) -> ConverseTurn:
    return ConverseTurn(
        message={"role": "assistant", "content": list(content)},
        stop_reason="tool_use" if any("toolUse" in item for item in content) else "end_turn",
        usage={"inputTokens": 10, "outputTokens": 5},
        metrics={},
    )


def wiki_page(
    *,
    title: str = "Test Article",
    page_type: str = "source",
    sources: tuple[str, ...] = ("raw/article.txt",),
    body: str = "Validated knowledge.",
) -> str:
    source_lines = "\n".join(f"  - {source}" for source in sources)
    source_bullets = "\n".join(f"- {source}" for source in sources)
    return f"""---
title: {title}
page_type: {page_type}
updated: 2026-08-06
sources:
{source_lines}
---
# {title}

{body}

## Sources

{source_bullets}
"""


class CoreTests(unittest.TestCase):
    def test_confidence_failure_fails_closed_without_service_error(self) -> None:
        class AnsweringAgent:
            def answer(self, question):
                return AnswerResult(
                    status="answered",
                    answer="Grounded answer",
                    citations=(),
                    usage={"inputTokens": 3},
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BedrockSettings(
                project_root=Path(temp_dir),
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            service = WikiService(
                settings,
                agent=AnsweringAgent(),
                confidence_evaluator=BrokenConfidenceEvaluator(),
            )

            with self.assertLogs("backend.app.service", level="WARNING"):
                answer = service.ask("Question")

        self.assertEqual(answer["status"], "insufficient_knowledge")
        self.assertIn("only answer from the available Wiki documents", answer["answer"])
        self.assertEqual(answer["citations"], [])
        self.assertIsNone(answer["confidence_score"])
        self.assertEqual(
            answer["debug"]["guardrail"]["reasons"],
            ["verification_unavailable"],
        )

    def test_unsupported_answer_is_replaced_and_citations_are_removed(self) -> None:
        class AnsweringAgent:
            def answer(self, question):
                return AnswerResult(
                    status="answered",
                    answer="An unsupported general-knowledge answer.",
                    citations=(Citation("concepts/example.md", ("raw/example.txt",)),),
                    usage={},
                    pages_read=("concepts/example.md",),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BedrockSettings(
                project_root=Path(temp_dir),
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            service = WikiService(
                settings,
                agent=AnsweringAgent(),
                confidence_evaluator=RejectingConfidenceEvaluator(),
            )
            answer = service.ask("General question")

        self.assertEqual(answer["status"], "insufficient_knowledge")
        self.assertNotIn("general-knowledge", answer["answer"])
        self.assertEqual(answer["citations"], [])
        self.assertEqual(answer["confidence_score"], 9.3)
        self.assertTrue(answer["debug"]["guardrail"]["applied"])
        self.assertIn(
            "unsupported_material_claim",
            answer["debug"]["guardrail"]["reasons"],
        )

    def test_existing_abstention_is_sanitized_in_question_language(self) -> None:
        class AbstainingAgent:
            def answer(self, question):
                return AnswerResult(
                    status="insufficient_knowledge",
                    answer="Non lo so, ma una pagina ha una data non pertinente.",
                    citations=(Citation("concepts/example.md", ("raw/example.txt",)),),
                    usage={},
                    pages_read=("concepts/example.md",),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BedrockSettings(
                project_root=Path(temp_dir),
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            service = WikiService(
                settings,
                agent=AbstainingAgent(),
                confidence_evaluator=RejectingConfidenceEvaluator(language="italian"),
            )
            answer = service.ask("Qual è la data di oggi?")

        self.assertEqual(answer["status"], "insufficient_knowledge")
        self.assertTrue(answer["answer"].startswith("Posso rispondere solo"))
        self.assertNotIn("data non pertinente", answer["answer"])
        self.assertEqual(answer["citations"], [])

    def test_index_ownership_is_unambiguous_and_step_default_is_24(self) -> None:
        schema = (Path(__file__).parents[1] / "AGENTS.md").read_text(encoding="utf-8-sig")
        settings = BedrockSettings(
            project_root=Path.cwd(),
            region_name="eu-west-1",
            bedrock_model_id="test-model",
        )

        self.assertIn("The application rebuilds it", schema)
        self.assertIn("must never write it", schema)
        self.assertNotIn("Update `index.md`", schema)
        self.assertEqual(settings.max_agent_steps, 24)

    def test_ingestion_step_limit_reports_recent_tool_names(self) -> None:
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "search-1",
                        "name": "search_wiki",
                        "input": {"query": "knowledge"},
                    }
                }
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            agent = WikiAgent(repository, ScriptedBedrock(turns), max_steps=1)

            with self.assertLogs("backend.app.agent", level="WARNING") as logs:
                with self.assertRaises(AgentValidationError) as raised:
                    agent.ingest("Ingest raw/article.txt into the wiki.")

        self.assertIn("Recent tools: search_wiki", str(raised.exception))
        self.assertIn("after 1 rounds", logs.output[0])

    def test_step_boundary_commits_only_valid_work_with_source_summary(self) -> None:
        page_content = """---
title: Test Article
page_type: source
updated: 2026-08-06
sources:
  - raw/article.txt
---
# Test Article

Validated knowledge.

## Sources

- raw/article.txt
"""
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "write-1",
                        "name": "write_wiki_page",
                        "input": {
                            "path": "sources/test-article.md",
                            "content": page_content,
                        },
                    }
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            agent = WikiAgent(repository, ScriptedBedrock(turns), max_steps=1)

            with self.assertLogs("backend.app.agent", level="WARNING"):
                result = agent.ingest("Ingest raw/article.txt into the wiki.")

            self.assertEqual(result.pages_written, ("sources/test-article.md",))
            self.assertIn("1-round ingestion boundary", result.message)
            self.assertTrue(repository.is_ingested("raw/article.txt"))

    def test_step_boundary_rejects_staged_work_without_source_summary(self) -> None:
        page_content = """---
title: Test Concept
page_type: concept
updated: 2026-08-06
sources:
  - raw/article.txt
---
# Test Concept

Validated knowledge.

## Sources

- raw/article.txt
"""
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "write-1",
                        "name": "write_wiki_page",
                        "input": {"path": "concepts/test.md", "content": page_content},
                    }
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            agent = WikiAgent(repository, ScriptedBedrock(turns), max_steps=1)

            with self.assertLogs("backend.app.agent", level="WARNING"):
                with self.assertRaises(AgentValidationError) as raised:
                    agent.ingest("Ingest raw/article.txt into the wiki.")

            self.assertIn("mandatory source-summary page", str(raised.exception))
            self.assertFalse((backend_root / "wiki" / "concepts" / "test.md").exists())

    def test_ingestion_tools_follow_discovery_and_write_phases(self) -> None:
        page_content = """---
title: Test Article
page_type: source
updated: 2026-08-06
sources:
  - raw/article.txt
---
# Test Article

Knowledge.

## Sources

- raw/article.txt
"""
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "search",
                        "name": "search_wiki",
                        "input": {"query": "knowledge"},
                    }
                }
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "write",
                        "name": "write_wiki_page",
                        "input": {
                            "path": "sources/test-article.md",
                            "content": page_content,
                        },
                    }
                }
            ),
            assistant_turn({"text": "Done."}),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            scripted = ScriptedBedrock(turns)
            agent = WikiAgent(repository, scripted, max_steps=8)

            result = agent.ingest("Ingest raw/article.txt into the wiki.")

        def tool_names(call):
            return {tool["toolSpec"]["name"] for tool in call["tools"]}

        self.assertIn("search_wiki", tool_names(scripted.calls[0]))
        self.assertNotIn("read_raw_source", tool_names(scripted.calls[0]))
        self.assertEqual(tool_names(scripted.calls[2]), {"write_wiki_page"})
        self.assertEqual(result.pages_written, ("sources/test-article.md",))

    def test_premature_completion_is_repaired_before_validation(self) -> None:
        page_content = """---
title: Test Article
page_type: source
updated: 2026-08-06
sources:
  - raw/article.txt
---
# Test Article

Knowledge.

## Sources

- raw/article.txt
"""
        turns = [
            assistant_turn({"text": "Done too early."}),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "write",
                        "name": "write_wiki_page",
                        "input": {
                            "path": "sources/test-article.md",
                            "content": page_content,
                        },
                    }
                }
            ),
            assistant_turn({"text": "Done."}),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            scripted = ScriptedBedrock(turns)
            agent = WikiAgent(repository, scripted, max_steps=8)

            result = agent.ingest("Ingest raw/article.txt into the wiki.")

        repair_messages = [
            block["text"]
            for call in scripted.calls
            for message in call["messages"]
            if message["role"] == "user"
            for block in message["content"]
            if "text" in block and block["text"].startswith("Ingestion is incomplete")
        ]
        self.assertTrue(any("source-summary" in text for text in repair_messages))
        self.assertEqual(result.pages_written, ("sources/test-article.md",))

    def test_configuration_repr_never_contains_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "aws_credentials.json").write_text(
                json.dumps(
                    {
                        "region_name": "eu-west-1",
                        "bedrock_model_id": "test-model",
                        "aws_access_key_id": "test-access",
                        "aws_secret_access_key": "do-not-print-this",
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(root, environ={})

            self.assertEqual(settings.credentials_source, "explicit")
            self.assertNotIn("do-not-print-this", repr(settings))
            self.assertNotIn("test-access", repr(settings))

    def test_repository_rejects_traversal_hidden_paths_and_missing_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            (backend_root / "raw").mkdir(parents=True)
            repository = WikiRepository(backend_root)

            with self.assertRaises(UnsafePathError):
                repository.read_raw("raw/../secret.txt")
            with self.assertRaises(UnsafePathError):
                repository.normalize_wiki_path(".hidden/page.md")
            with self.assertRaises(RepositoryError):
                repository.write_wiki_pages({"sources/page.md": "# Missing frontmatter"})

    def test_failed_log_reference_does_not_mark_source_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            wiki_root = backend_root / "wiki"
            raw_root.mkdir(parents=True)
            wiki_root.mkdir()
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            (wiki_root / "log.md").write_text(
                "## ingest | raw/article.txt\n\n- Status: failed\n", encoding="utf-8"
            )

            repository = WikiRepository(backend_root)

            self.assertFalse(repository.is_ingested("raw/article.txt"))

    def test_repository_enforces_schema_sources_and_local_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)

            with self.assertRaisesRegex(RepositoryError, "page_type"):
                repository.write_wiki_pages(
                    {
                        "sources/wrong-type.md": wiki_page(
                            page_type="concept", title="Wrong Type"
                        )
                    }
                )
            with self.assertRaisesRegex(RepositoryError, "exact '## Sources'"):
                repository.write_wiki_pages(
                    {
                        "sources/no-sources-heading.md": wiki_page().replace(
                            "## Sources", "## Provenance"
                        )
                    }
                )
            with self.assertRaisesRegex(RepositoryError, "Broken wiki link"):
                repository.write_wiki_pages(
                    {
                        "sources/broken-link.md": wiki_page(
                            body="See [missing](../concepts/missing.md)."
                        )
                    }
                )

    def test_multi_page_write_rolls_back_every_file_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            original_atomic_write = repository._atomic_write
            calls = 0

            def fail_second_write(path: Path, content: str) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated disk failure")
                original_atomic_write(path, content)

            with patch.object(repository, "_atomic_write", side_effect=fail_second_write):
                with self.assertRaisesRegex(RepositoryError, "rolled back"):
                    repository.write_wiki_pages(
                        {
                            "concepts/concept.md": wiki_page(
                                title="Concept", page_type="concept"
                            ),
                            "sources/article.md": wiki_page(),
                        }
                    )

            self.assertFalse((backend_root / "wiki" / "concepts" / "concept.md").exists())
            self.assertFalse((backend_root / "wiki" / "sources" / "article.md").exists())
            self.assertFalse((backend_root / "wiki" / "index.md").exists())

    def test_source_content_change_makes_ingestion_pending_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            source = raw_root / "article.txt"
            source.write_text("Version one", encoding="utf-8")
            repository = WikiRepository(backend_root)

            repository.commit_ingestion(
                "raw/article.txt", {"sources/article.md": wiki_page()}
            )
            self.assertTrue(repository.is_ingested("raw/article.txt"))

            source.write_text("Version two", encoding="utf-8")
            self.assertFalse(repository.is_ingested("raw/article.txt"))
            self.assertEqual(repository.list_raw_documents()[0].status, "Pending")

    def test_log_failure_does_not_turn_a_committed_ingestion_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)

            class CommittingAgent:
                def ingest(self, prompt: str) -> IngestionResult:
                    pages = repository.commit_ingestion(
                        "raw/article.txt", {"sources/article.md": wiki_page()}
                    )
                    return IngestionResult(
                        source_path="raw/article.txt",
                        prompt=prompt,
                        pages_written=tuple(pages),
                        message="Committed",
                        usage={},
                    )

            settings = BedrockSettings(
                project_root=project_root,
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            service = WikiService(settings, repository=repository, agent=CommittingAgent())
            with patch.object(
                repository, "append_log", side_effect=RepositoryError("log unavailable")
            ):
                result = service.update_wiki(["article.txt"])

            self.assertEqual(result["summary"]["processed"], 1)
            self.assertEqual(result["summary"]["failed"], 0)
            self.assertIn("warning", result["processed"][0])
            self.assertTrue(repository.is_ingested("raw/article.txt"))

    def test_answer_requires_research_results_before_submission(self) -> None:
        page = wiki_page(body="The answer is 42.")
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "read",
                        "name": "read_wiki_page",
                        "input": {"path": "sources/article.md"},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "early-answer",
                        "name": "submit_answer",
                        "input": {
                            "status": "answered",
                            "answer": "42",
                            "citations": [
                                {
                                    "wiki_path": "sources/article.md",
                                    "source_paths": ["raw/article.txt"],
                                }
                            ],
                        },
                    }
                },
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "answer",
                        "name": "submit_answer",
                        "input": {
                            "status": "answered",
                            "answer": "42",
                            "citations": [
                                {
                                    "wiki_path": "sources/article.md",
                                    "source_paths": ["raw/article.txt"],
                                }
                            ],
                        },
                    }
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("42", encoding="utf-8")
            repository = WikiRepository(backend_root)
            repository.write_wiki_pages({"sources/article.md": page})
            scripted = ScriptedBedrock(turns)
            result = WikiAgent(repository, scripted, max_steps=4).answer("What is it?")

        self.assertEqual(result.answer, "42")
        self.assertIn(
            "must be the only tool call",
            str(scripted.calls[1]["messages"]),
        )

    def test_answer_citation_must_include_all_page_provenance(self) -> None:
        page = wiki_page(sources=("raw/article.txt", "raw/second.txt"))
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "read",
                        "name": "read_wiki_page",
                        "input": {"path": "sources/article.md"},
                    }
                }
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "incomplete",
                        "name": "submit_answer",
                        "input": {
                            "status": "answered",
                            "answer": "Grounded",
                            "citations": [
                                {
                                    "wiki_path": "sources/article.md",
                                    "source_paths": ["raw/article.txt"],
                                }
                            ],
                        },
                    }
                }
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "complete",
                        "name": "submit_answer",
                        "input": {
                            "status": "answered",
                            "answer": "Grounded",
                            "citations": [
                                {
                                    "wiki_path": "sources/article.md",
                                    "source_paths": ["raw/article.txt", "raw/second.txt"],
                                }
                            ],
                        },
                    }
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("One", encoding="utf-8")
            (raw_root / "second.txt").write_text("Two", encoding="utf-8")
            repository = WikiRepository(backend_root)
            repository.write_wiki_pages({"sources/article.md": page})
            result = WikiAgent(repository, ScriptedBedrock(turns), max_steps=5).answer(
                "What is grounded?"
            )

        self.assertEqual(
            result.citations[0].source_paths,
            ("raw/article.txt", "raw/second.txt"),
        )

    def test_graph_lint_and_safe_bidirectional_link_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            repository.write_wiki_pages(
                {
                    "sources/article.md": wiki_page(body="A source summary."),
                    "concepts/topic.md": wiki_page(
                        title="Topic", page_type="concept", body="A reusable topic."
                    ),
                }
            )

            before = repository.lint_wiki()
            self.assertTrue(before["valid"])
            self.assertEqual(before["graph"]["isolated"], 2)
            self.assertEqual(
                {issue["code"] for issue in before["issues"]}, {"isolated_page"}
            )

            result = repository.apply_cross_links(
                [("sources/article.md", "concepts/topic.md")]
            )

            self.assertEqual(len(result["pairs_added"]), 1)
            self.assertEqual(
                result["pages_updated"], ["concepts/topic.md", "sources/article.md"]
            )
            source_content = repository.read_wiki_page("sources/article.md")
            concept_content = repository.read_wiki_page("concepts/topic.md")
            self.assertIn("A source summary.", source_content)
            self.assertIn("A reusable topic.", concept_content)
            self.assertIn("[Topic](<../concepts/topic.md>)", source_content)
            self.assertIn("[Test Article](<../sources/article.md>)", concept_content)
            after = repository.lint_wiki()
            self.assertEqual(after["graph"]["links"], 2)
            self.assertEqual(after["graph"]["isolated"], 0)
            self.assertEqual(after["issues"], [])

    def test_semantic_link_agent_reads_both_pages_before_safe_repair(self) -> None:
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "read-source",
                        "name": "read_wiki_page",
                        "input": {"path": "sources/article.md"},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "read-topic",
                        "name": "read_wiki_page",
                        "input": {"path": "concepts/topic.md"},
                    }
                },
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "submit-links",
                        "name": "submit_link_repairs",
                        "input": {
                            "links": [
                                {
                                    "source_path": "sources/article.md",
                                    "target_path": "concepts/topic.md",
                                    "reason": "The concept is directly explained by the source.",
                                }
                            ]
                        },
                    }
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_root = Path(temp_dir) / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (raw_root / "article.txt").write_text("Knowledge", encoding="utf-8")
            repository = WikiRepository(backend_root)
            repository.write_wiki_pages(
                {
                    "sources/article.md": wiki_page(body="A source summary."),
                    "concepts/topic.md": wiki_page(
                        title="Topic", page_type="concept", body="A reusable topic."
                    ),
                }
            )
            scripted = ScriptedBedrock(turns)
            result = WikiAgent(repository, scripted, max_steps=4).repair_links(max_links=3)

            self.assertEqual(len(result.links_added), 1)
            self.assertEqual(result.graph_before["isolated"], 2)
            self.assertEqual(result.graph_after["isolated"], 0)
            self.assertEqual(
                result.pages_updated, ("concepts/topic.md", "sources/article.md")
            )
            self.assertIn(
                "[Topic](<../concepts/topic.md>)",
                repository.read_wiki_page("sources/article.md"),
            )

    def test_mocked_ingestion_and_grounded_question_end_to_end(self) -> None:
        page_content = """---
title: Test Article
page_type: source
updated: 2026-07-22
sources:
  - raw/article.txt
---
# Test Article

The source says the answer is 42.

## Sources

- raw/article.txt
"""
        turns = [
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "write-1",
                        "name": "write_wiki_page",
                        "input": {"path": "sources/test-article.md", "content": page_content},
                    }
                }
            ),
            assistant_turn({"text": "Integrated the article into the wiki."}),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "page-1",
                        "name": "read_wiki_page",
                        "input": {"path": "sources/test-article.md"},
                    }
                }
            ),
            assistant_turn(
                {
                    "toolUse": {
                        "toolUseId": "answer-1",
                        "name": "submit_answer",
                        "input": {
                            "status": "answered",
                            "answer": "The answer is 42.",
                            "citations": [
                                {
                                    "wiki_path": "sources/test-article.md",
                                    "source_paths": ["raw/article.txt"],
                                }
                            ],
                        },
                    }
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            raw_root = backend_root / "raw"
            raw_root.mkdir(parents=True)
            (backend_root / "AGENTS.md").write_text("Maintain a cited wiki.", encoding="utf-8")
            source_path = raw_root / "article.txt"
            source_path.write_text("The answer is 42.", encoding="utf-8")

            settings = BedrockSettings(
                project_root=project_root,
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            repository = WikiRepository(backend_root)
            scripted_bedrock = ScriptedBedrock(turns)
            agent = WikiAgent(repository, scripted_bedrock, max_steps=8)
            service = WikiService(
                settings,
                repository=repository,
                agent=agent,
                confidence_evaluator=StaticConfidenceEvaluator(),
            )

            before = service.list_documents()
            update = service.update_wiki(["article.txt"])
            answer = service.ask("What is the answer?")

            self.assertEqual(before[0]["relative_path"], "article.txt")
            self.assertEqual(before[0]["status"], "Pending")
            self.assertEqual(update["summary"]["processed"], 1)
            self.assertEqual(
                update["processed"][0]["prompt"],
                "Ingest raw/article.txt into the wiki.",
            )
            self.assertEqual(source_path.read_text(encoding="utf-8"), "The answer is 42.")
            self.assertTrue(repository.is_ingested("raw/article.txt"))
            self.assertTrue((backend_root / "wiki" / "index.md").is_file())
            self.assertIn("Status: success", (backend_root / "wiki" / "log.md").read_text())
            self.assertEqual(answer["answer"], "The answer is 42.")
            self.assertEqual(answer["confidence_score"], 8.6)
            self.assertEqual(answer["usage"]["inputTokens"], 22)
            self.assertEqual(answer["citations"][0]["source_paths"], ["raw/article.txt"])
            first_user_text = scripted_bedrock.calls[0]["messages"][0]["content"][0]["text"]
            self.assertEqual(first_user_text, "Ingest raw/article.txt into the wiki.")
            loaded_source = scripted_bedrock.calls[0]["messages"][0]["content"][1]["text"]
            self.assertIn('<raw_source path="raw/article.txt">', loaded_source)
            self.assertIn("The answer is 42.", loaded_source)

    def test_bedrock_error_exposes_only_safe_error_code(self) -> None:
        class AccessDenied(Exception):
            response = {"Error": {"Code": "AccessDeniedException", "Message": "private"}}

        class FailingClient:
            def converse(self, **kwargs):
                raise AccessDenied("do-not-return-private-error")

        settings = BedrockSettings(
            project_root=Path.cwd(),
            region_name="eu-west-1",
            bedrock_model_id="test-model",
        )
        client = BedrockConverseClient(settings, client=FailingClient())

        with self.assertRaises(BedrockError) as raised:
            client.converse(
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                system_prompt="Test",
            )

        self.assertIn("AccessDeniedException", str(raised.exception))
        self.assertNotIn("private", str(raised.exception))

    def test_transient_bedrock_validation_error_is_retried_once(self) -> None:
        class ValidationFailure(Exception):
            response = {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "provider-private-detail",
                }
            }

        class FlakyClient:
            def __init__(self) -> None:
                self.calls = 0

            def converse(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValidationFailure("provider-private-detail")
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "Recovered"}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {},
                }

        settings = BedrockSettings(
            project_root=Path.cwd(),
            region_name="eu-west-1",
            bedrock_model_id="test-model",
        )
        flaky = FlakyClient()
        client = BedrockConverseClient(settings, client=flaky)

        turn = client.converse(
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            system_prompt="Test",
        )

        self.assertEqual(flaky.calls, 2)
        self.assertEqual(turn.message["content"][0]["text"], "Recovered")


if __name__ == "__main__":
    unittest.main()
