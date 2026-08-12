from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.agent import AnswerResult, Citation, IngestionResult
from backend.app.answer_fixes import AnswerFixPlan
from backend.app.confidence import ConfidenceEvaluation
from backend.app.config import BedrockSettings
from backend.app.manager_actions import (
    ManagerActionContext,
    ManagerActionProposal,
    ManagerActionStore,
)
from backend.app.repository import WikiRepository
from backend.app.service import WikiService


def action(action_type: str) -> ManagerActionProposal:
    return ManagerActionProposal(
        action_id=f"{action_type}-123",
        action_type=action_type,
        subject="Generic operating procedure",
        previous_value="Old procedure" if action_type != "add_knowledge" else "",
        new_value="Use the documented approved procedure.",
        scope="Example organization",
        effective_period="",
        reason="Manager-approved POC action",
        needs_clarification=False,
        clarification_question="",
        language="english",
        usage={},
    )


class AcceptedConfidence:
    def evaluate(self, question, result):
        return ConfidenceEvaluation(
            score=9.4,
            usage={},
            claim_support=1.0,
            question_coverage=1.0,
            source_consistency=1.0,
            evidence_quality=1.0,
            abstention_score=1.0,
            has_unsupported_material_claim=False,
            has_unexplained_conflict=False,
            response_language="english",
        )


class WrongAnswerAgent:
    def answer(self, question):
        return AnswerResult(
            status="answered",
            answer="The incorrect ordering.",
            citations=(Citation("concepts/procedure.md", ("raw/procedure.md",)),),
            usage={},
            pages_read=("concepts/procedure.md",),
        )


class FixedInterpreter:
    def __init__(self, proposal):
        self.proposal = proposal

    @staticmethod
    def looks_like_action(message):
        return message.casefold().startswith(("fix answer", "add knowledge", "update knowledge"))

    @staticmethod
    def is_confirmation(message):
        return message.casefold().strip() == "confirm"

    @staticmethod
    def is_cancellation(message):
        return message.casefold().strip() == "cancel"

    def interpret(self, context, message, *, draft=None):
        return self.proposal


class FixedReviewer:
    def prepare(self, context, proposal):
        return AnswerFixPlan(
            supported=True,
            answer="Use the documented approved procedure.",
            citations=(Citation("concepts/procedure.md", ("raw/procedure.md",)),),
            target_page="concepts/procedure.md",
            failure_stage="generation",
            explanation="The previous answer reordered existing evidence.",
            usage={},
        )


class UnsupportedReviewer:
    def prepare(self, context, proposal):
        return AnswerFixPlan(
            supported=False,
            answer="",
            citations=(),
            target_page="",
            failure_stage="unknown",
            explanation="The existing Wiki does not support the proposed correction.",
            usage={},
        )


class DisabledSearch:
    enabled = False


class FakeKnowledgeStore:
    def __init__(self):
        self.calls = []

    def persist_knowledge(self, proposal, context):
        self.calls.append((proposal, context))
        return f"raw/manager-actions/{proposal.action_id}.md"


class RecordingKnowledgeAgent:
    def __init__(self):
        self.calls = []

    def answer(self, question):
        return WrongAnswerAgent().answer(question)

    def update_existing_knowledge(self, source_path, *, writable_pages):
        self.calls.append((source_path, writable_pages))
        return IngestionResult(
            source_path=source_path,
            prompt="Update existing knowledge",
            pages_written=writable_pages,
            message="Updated existing knowledge",
            usage={},
        )


class ManagerActionWorkflowTests(unittest.TestCase):
    def _repository(self, root: str) -> WikiRepository:
        backend = Path(root) / "backend"
        raw = backend / "raw"
        raw.mkdir(parents=True)
        (raw / "procedure.md").write_text("Approved procedure evidence.", encoding="utf-8")
        repository = WikiRepository(backend)
        repository.write_wiki_pages(
            {
                "concepts/procedure.md": """---
title: Procedure
page_type: concept
updated: 2026-08-11
sources:
- raw/procedure.md
---

# Procedure

The source contains the approved procedure.

## Sources

- raw/procedure.md
"""
            }
        )
        return repository

    def _service(self, root, repository, proposal, *, store=None, reviewer=None):
        settings = BedrockSettings(
            project_root=Path(root),
            region_name="eu-west-1",
            bedrock_model_id="test-model",
        )
        return WikiService(
            settings,
            repository=repository,
            agent=WrongAnswerAgent(),
            searcher=DisabledSearch(),
            confidence_evaluator=AcceptedConfidence(),
            correction_interpreter=FixedInterpreter(proposal),
            correction_store=store or ManagerActionStore(repository),
            answer_fix_reviewer=reviewer,
        )

    def test_fix_answer_requires_preview_then_updates_derived_wiki_without_raw_source(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            service = self._service(
                root,
                repository,
                action("fix_answer"),
                reviewer=FixedReviewer(),
            )
            service.ask("What is the procedure?", session_id="manager-fix")
            before = repository.read_wiki_page("concepts/procedure.md")
            proposed = service.ask(
                "Fix answer: use the documented approved procedure.",
                session_id="manager-fix",
            )
            self.assertEqual(proposed["status"], "manager_action_proposed")
            self.assertFalse(proposed["manager_action"]["changes_knowledge"])
            self.assertEqual(repository.read_wiki_page("concepts/procedure.md"), before)

            applied = service.ask("Confirm", session_id="manager-fix")
            after = repository.read_wiki_page("concepts/procedure.md")
            raw_actions = list((repository.raw_root / "manager-actions").glob("*.md"))
            feedback = list((repository.backend_root / "feedback" / "answer-fixes").glob("*.json"))

        self.assertEqual(applied["status"], "manager_action_applied")
        self.assertIn("Manager-reviewed guidance", after)
        self.assertIn("Use the documented approved procedure.", after)
        self.assertEqual(raw_actions, [])
        self.assertEqual(len(feedback), 1)
        self.assertEqual(applied["confidence_score"], 9.4)

    def test_unsupported_fix_is_rejected_without_any_persistent_change(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            service = self._service(
                root,
                repository,
                action("fix_answer"),
                reviewer=UnsupportedReviewer(),
            )
            service.ask("What is the procedure?", session_id="manager-unsupported")
            before = repository.read_wiki_page("concepts/procedure.md")
            service.ask(
                "Fix answer: use a different unsupported procedure.",
                session_id="manager-unsupported",
            )
            result = service.ask("Confirm", session_id="manager-unsupported")
            raw_actions = list((repository.raw_root / "manager-actions").glob("*.md"))
            feedback_root = repository.backend_root / "feedback" / "answer-fixes"

            self.assertEqual(result["status"], "manager_action_failed")
            self.assertEqual(repository.read_wiki_page("concepts/procedure.md"), before)
            self.assertEqual(raw_actions, [])
            self.assertFalse(feedback_root.exists())

    def test_add_knowledge_can_start_without_previous_answer_and_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = FakeKnowledgeStore()
            service = self._service(root, repository, action("add_knowledge"), store=store)
            proposed = service.ask(
                "Add knowledge: use the documented approved procedure.",
                session_id="manager-add",
            )
            self.assertEqual(proposed["status"], "manager_action_proposed")
            self.assertTrue(proposed["manager_action"]["changes_knowledge"])
            self.assertEqual(store.calls, [])
            update_result = {
                "processed": [
                    {
                        "source_path": "raw/manager-actions/add_knowledge-123.md",
                        "pages_written": [],
                        "usage": {},
                    }
                ],
                "skipped": [],
                "failed": [],
            }
            with patch.object(service, "update_wiki", return_value=update_result) as update:
                applied = service.ask("Confirm", session_id="manager-add")

        self.assertEqual(applied["status"], "manager_action_applied")
        self.assertEqual(len(store.calls), 1)
        update.assert_called_once_with(["raw/manager-actions/add_knowledge-123.md"])

    def test_add_and_update_sources_have_distinct_history_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = ManagerActionStore(repository)
            context = ManagerActionContext(question="", answer="", citations=())

            add_path = store.persist_knowledge(action("add_knowledge"), context)
            update_path = store.persist_knowledge(action("update_knowledge"), context)
            add_content = repository.read_raw(add_path)
            update_content = repository.read_raw(update_path)

            self.assertNotIn("## Superseded value", add_content)
            self.assertIn("## Superseded value", update_content)
            self.assertIn("Old procedure", update_content)

    def test_update_routes_to_existing_canonical_page_without_normal_ingestion(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            action_root = repository.raw_root / "manager-actions"
            action_root.mkdir(parents=True)
            (action_root / "update_knowledge-123.md").write_text(
                "Approved updated procedure.", encoding="utf-8"
            )
            agent = RecordingKnowledgeAgent()
            service = self._service(
                root,
                repository,
                action("update_knowledge"),
                store=FakeKnowledgeStore(),
            )
            service._agent = agent

            context = ManagerActionContext(
                question="What is the procedure?",
                answer="The old procedure.",
                citations=(
                    {
                        "wiki_path": "concepts/procedure.md",
                        "source_paths": ["raw/procedure.md"],
                    },
                ),
            )
            result = service._update_existing_knowledge(
                "raw/manager-actions/update_knowledge-123.md",
                proposal=action("update_knowledge"),
                context=context,
            )

        self.assertEqual(result["failed"], [])
        self.assertEqual(
            agent.calls,
            [
                (
                    "raw/manager-actions/update_knowledge-123.md",
                    ("concepts/procedure.md",),
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
