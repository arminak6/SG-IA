from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from backend.app.agent import AnswerResult, Citation, IngestionResult
from backend.app.answer_fixes import AnswerFixPlan
from backend.app.confidence import ConfidenceEvaluation
from backend.app.config import BedrockSettings
from backend.app.manager_actions import (
    ManagerActionContext,
    ManagerActionInterpreter,
    ManagerActionProposal,
    ManagerActionStore,
    ManagerKnowledgeWrite,
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
        return message.casefold().startswith(("/fix", "/add", "/update"))

    @staticmethod
    def is_confirmation(message):
        return message.casefold().strip().lstrip("/") == "confirm"

    @staticmethod
    def is_cancellation(message):
        return message.casefold().strip().lstrip("/") == "cancel"

    @staticmethod
    def is_control_command(message):
        return message.casefold().strip() in {"/confirm", "/cancel"}

    @staticmethod
    def looks_like_unmarked_action(message):
        lowered = message.casefold()
        return lowered.startswith("for ") and " replace " in lowered and " with " in lowered

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


class ReversibleKnowledgeStore(FakeKnowledgeStore):
    def __init__(self):
        super().__init__()
        self.rollbacks = []

    def persist_knowledge(self, proposal, context):
        self.calls.append((proposal, context))
        return ManagerKnowledgeWrite(
            "raw/manager-knowledge/procedure.md",
            "Previous stable source\n",
        )

    def rollback_knowledge(self, write):
        self.rollbacks.append(write)


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

## Related notes

This post-sources section must be preserved by answer fixes.
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
                "/fix Use the documented approved procedure.",
                session_id="manager-fix",
            )
            self.assertEqual(proposed["status"], "manager_action_proposed")
            self.assertFalse(proposed["manager_action"]["changes_knowledge"])
            self.assertEqual(repository.read_wiki_page("concepts/procedure.md"), before)

            applied = service.ask("/confirm", session_id="manager-fix")
            after = repository.read_wiki_page("concepts/procedure.md")
            raw_actions = list((repository.raw_root / "manager-actions").glob("*.md"))
            feedback = list((repository.backend_root / "feedback" / "answer-fixes").glob("*.json"))

        self.assertEqual(applied["status"], "manager_action_applied")
        self.assertIn("Manager-reviewed guidance", after)
        self.assertIn("Use the documented approved procedure.", after)
        self.assertIn("## Related notes", after)
        self.assertIn("This post-sources section must be preserved", after)
        self.assertEqual(raw_actions, [])
        self.assertEqual(len(feedback), 1)
        self.assertEqual(applied["confidence_score"], 9.4)

    def test_unsupported_fix_becomes_update_proposal_without_persistent_change(self):
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
                "/fix Use a different unsupported procedure.",
                session_id="manager-unsupported",
            )
            result = service.ask("/confirm", session_id="manager-unsupported")
            raw_actions = list((repository.raw_root / "manager-actions").glob("*.md"))
            feedback_root = repository.backend_root / "feedback" / "answer-fixes"

            self.assertEqual(result["status"], "manager_action_proposed")
            self.assertEqual(
                result["manager_action"]["action_type"],
                "update_knowledge",
            )
            self.assertTrue(result["manager_action"]["changes_knowledge"])
            self.assertIn("new confirmation is required", result["answer"])
            self.assertEqual(repository.read_wiki_page("concepts/procedure.md"), before)
            self.assertEqual(raw_actions, [])
            self.assertFalse(feedback_root.exists())

    def test_converted_update_requires_second_confirmation_before_apply(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = ReversibleKnowledgeStore()
            service = self._service(
                root,
                repository,
                action("fix_answer"),
                store=store,
                reviewer=UnsupportedReviewer(),
            )
            service.ask("What is the procedure?", session_id="manager-convert")
            service.ask(
                "/fix Use a different manager-approved procedure.",
                session_id="manager-convert",
            )

            converted = service.ask("/confirm", session_id="manager-convert")
            self.assertEqual(converted["status"], "manager_action_proposed")
            self.assertEqual(store.calls, [])

            successful_update = {
                "processed": [
                    {
                        "source_path": "raw/manager-knowledge/procedure.md",
                        "pages_written": ["concepts/procedure.md"],
                        "usage": {},
                    }
                ],
                "skipped": [],
                "failed": [],
            }
            with patch.object(
                service,
                "_update_existing_knowledge",
                return_value=successful_update,
            ):
                applied = service.ask("/confirm", session_id="manager-convert")

            self.assertEqual(applied["status"], "manager_action_applied")
            self.assertEqual(len(store.calls), 1)
            self.assertEqual(store.calls[0][0].action_type, "update_knowledge")

    def test_answer_context_loads_cited_manager_snapshot_for_update_merge(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            source_path, _ = repository.write_manager_knowledge_source(
                "meeting.md",
                """# Manager Knowledge: Meeting

## Current approved knowledge

The meeting is held annually on 13 July.
""",
            )
            service = self._service(root, repository, action("update_knowledge"))

            service._remember_answer(
                "manager-snapshot",
                "When is the meeting?",
                {
                    "answer": "The meeting is held annually on 13 July.",
                    "citations": [
                        {
                            "wiki_path": "entities/meeting.md",
                            "source_paths": [source_path],
                        }
                    ],
                },
            )
            context = service._session("manager-snapshot").context

        self.assertEqual(
            context.maintained_knowledge,
            (
                {
                    "source_path": "raw/manager-knowledge/meeting.md",
                    "current_value": "The meeting is held annually on 13 July.",
                    "subject": "Meeting",
                    "scope": "",
                    "effective_period": "",
                },
            ),
        )

    def test_add_knowledge_can_start_without_previous_answer_and_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = FakeKnowledgeStore()
            service = self._service(root, repository, action("add_knowledge"), store=store)
            proposed = service.ask(
                "/add Use the documented approved procedure.",
                session_id="manager-add",
            )

            self.assertNotIn("Previous value:", proposed["answer"])
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
                applied = service.ask("/confirm", session_id="manager-add")

        self.assertEqual(applied["status"], "manager_action_applied")
        self.assertEqual(len(store.calls), 1)
        update.assert_called_once_with(
            ["raw/manager-actions/add_knowledge-123.md"],
            allow_manager_knowledge=True,
        )

    def test_short_add_creates_disposable_manager_source_and_temp_state_is_removed(self):
        temporary_root: Path | None = None
        with tempfile.TemporaryDirectory() as root:
            temporary_root = Path(root)
            repository = self._repository(root)
            short_add = replace(
                action("add_knowledge"),
                subject="Support desk hours",
                new_value="The support desk closes at 18:00.",
                scope="all offices",
            )
            service = self._service(
                root,
                repository,
                short_add,
                store=ManagerActionStore(repository),
            )
            proposed = service.ask(
                "/add The support desk closes at 18:00 for all offices.",
                session_id="manager-short-add",
            )

            def successful_update(paths, *, allow_manager_knowledge=False):
                return {
                    "processed": [
                        {
                            "source_path": paths[0],
                            "pages_written": [],
                            "usage": {},
                        }
                    ],
                    "skipped": [],
                    "failed": [],
                }

            with patch.object(service, "update_wiki", side_effect=successful_update):
                applied = service.ask("/confirm", session_id="manager-short-add")

            sources = list((repository.raw_root / "manager-knowledge").glob("*.md"))
            self.assertEqual(proposed["status"], "manager_action_proposed")
            self.assertEqual(applied["status"], "manager_action_applied")
            self.assertEqual(len(sources), 1)
            self.assertIn("support desk", sources[0].read_text(encoding="utf-8").casefold())

        self.assertIsNotNone(temporary_root)
        self.assertFalse(temporary_root.exists())

    def test_new_question_after_answer_bypasses_manager_action_interpreter(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = FakeKnowledgeStore()
            service = self._service(
                root,
                repository,
                action("update_knowledge"),
                store=store,
            )
            service.ask("When is the procedure used?", session_id="manager-context")

            answer = service.ask(
                "What Fondo Est limits are indicated for maternity, glasses/lenses and physiotherapy?",
                session_id="manager-context",
            )

        self.assertEqual(answer["status"], "answered")
        self.assertIsNone(answer.get("manager_action"))
        self.assertEqual(store.calls, [])

    def test_unmarked_replacement_gets_manager_form_guidance_instead_of_qa(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            service = self._service(
                root,
                repository,
                action("update_knowledge"),
            )
            service.ask("What is the current policy?", session_id="manager-guidance")

            result = service.ask(
                'For the policy, replace "the old value" with "the new value".',
                session_id="manager-guidance",
            )

        self.assertEqual(result["status"], "manager_action_command_required")
        self.assertIn("click **update knowledge**", result["answer"].casefold())

    def test_add_then_update_reuses_one_stable_manager_source(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = ManagerActionStore(repository)
            context = ManagerActionContext(question="", answer="", citations=())

            add_write = store.persist_knowledge(action("add_knowledge"), context)
            repository.write_wiki_pages(
                {
                    "concepts/procedure.md": f"""---
title: Procedure
page_type: concept
updated: 2026-08-11
sources:
- raw/procedure.md
- {add_write.source_path}
---

# Procedure

The source contains the approved procedure.

## Sources

- raw/procedure.md
- {add_write.source_path}
"""
                }
            )
            linked_context = ManagerActionContext(
                question="What is the procedure?",
                answer="Use the documented approved procedure.",
                citations=(
                    {
                        "wiki_path": "concepts/procedure.md",
                        "source_paths": ["raw/procedure.md", add_write.source_path],
                    },
                ),
            )
            update_write = store.persist_knowledge(
                replace(
                    action("update_knowledge"),
                    subject="Approved procedure instruction",
                    new_value=(
                        "Use the revised approved procedure and retain the documented "
                        "approval step."
                    ),
                    manager_input="Also retain the documented approval step.",
                ),
                linked_context,
            )
            update_content = repository.read_raw(update_write.source_path)

            self.assertEqual(add_write.source_path, update_write.source_path)
            self.assertEqual(
                update_write.source_path,
                "raw/manager-knowledge/generic-operating-procedure.md",
            )
            self.assertIn(
                "Use the revised approved procedure and retain the documented approval step.",
                update_content,
            )
            self.assertNotIn("Also retain", update_content)
            self.assertNotIn("Old procedure", update_content)
            self.assertNotIn("Superseded value", update_content)
            self.assertEqual(len(repository.list_raw_documents()), 2)

    def test_store_rollback_restores_previous_stable_value(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = ManagerActionStore(repository)
            context = ManagerActionContext(question="", answer="", citations=())
            first = store.persist_knowledge(action("add_knowledge"), context)
            first_content = repository.read_raw(first.source_path)
            second = store.persist_knowledge(
                replace(
                    action("update_knowledge"),
                    new_value="Use the replacement procedure.",
                ),
                context,
            )

            store.rollback_knowledge(second)

            self.assertEqual(repository.read_raw(first.source_path), first_content)
            self.assertEqual(len(repository.list_raw_documents()), 2)

    def test_failed_integration_rolls_back_the_stable_source(self):
        with tempfile.TemporaryDirectory() as root:
            repository = self._repository(root)
            store = ReversibleKnowledgeStore()
            service = self._service(
                root,
                repository,
                action("update_knowledge"),
                store=store,
            )
            service.ask("What is the procedure?", session_id="manager-rollback")
            service.ask(
                "/update Use the replacement procedure.",
                session_id="manager-rollback",
            )
            failed_update = {
                "processed": [],
                "skipped": [],
                "failed": [
                    {
                        "source_path": "raw/manager-knowledge/procedure.md",
                        "error": "integration failed",
                    }
                ],
            }
            with patch.object(
                service,
                "_update_existing_knowledge",
                return_value=failed_update,
            ):
                result = service.ask("/confirm", session_id="manager-rollback")

            pending = service._session("manager-rollback").pending

        self.assertEqual(result["status"], "manager_action_failed")
        self.assertEqual(len(store.rollbacks), 1)
        self.assertIsNotNone(pending)
        self.assertIsNone(pending.source_path)

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

    def test_update_preserves_manager_metadata_unless_explicitly_changed(self):
        proposal = replace(
            action("update_knowledge"),
            subject="Procedure timing",
            scope="model-inferred scope",
            effective_period="ongoing",
            manager_input="Change the confirmation lead time from one day to two days.",
        )
        context = ManagerActionContext(
            question="When is it confirmed?",
            answer="One day beforehand.",
            citations=(),
            maintained_knowledge=(
                {
                    "source_path": "raw/manager-knowledge/procedure.md",
                    "current_value": "One day beforehand.",
                    "subject": "Generic operating procedure",
                    "scope": "operations",
                    "effective_period": "weekly",
                },
            ),
        )

        completed = ManagerActionInterpreter._complete_explicit_proposal(
            proposal,
            context=context,
        )

        self.assertEqual(completed.subject, "Generic operating procedure")
        self.assertEqual(completed.scope, "operations")
        self.assertEqual(completed.effective_period, "weekly")


if __name__ == "__main__":
    unittest.main()
