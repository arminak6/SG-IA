from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from backend.app.agent import AnswerResult, Citation
from backend.app.bedrock import ConverseTurn
from backend.app.confidence import ConfidenceEvaluation
from backend.app.config import BedrockSettings
from backend.app.manager_actions import (
    ManagerActionContext,
    ManagerActionInterpreter,
    ManagerActionProposal,
    ManagerActionStore,
)
from backend.app.repository import RepositoryError, WikiRepository
from backend.app.service import WikiService


class ScriptedBedrock:
    def __init__(self, *turns: ConverseTurn) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("Unexpected Bedrock call")
        return self.turns.pop(0)


def correction_turn(*, needs_clarification: bool = False) -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "correction-1",
                        "name": "submit_manager_action",
                        "input": {
                            "is_manager_action": True,
                            "action_type": "update_knowledge",
                            "subject": "Company holiday start date",
                            "previous_value": "28 December",
                            "new_value": "27 December",
                            "scope": "Sinergia company holiday",
                            "effective_period": "" if needs_clarification else "2026",
                            "reason": "Manager supplied the corrected date.",
                            "needs_clarification": needs_clarification,
                            "clarification_question": (
                                "Which year does this correction apply to?"
                                if needs_clarification
                                else ""
                            ),
                            "language": "english",
                        },
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 20, "outputTokens": 10},
        metrics={},
    )


def proposal(
    *,
    action_type: str = "update_knowledge",
    needs_clarification: bool = False,
) -> ManagerActionProposal:
    return ManagerActionProposal(
        action_id="correction123",
        action_type=action_type,
        subject="Company holiday start date",
        previous_value="" if action_type == "add_knowledge" else "28 December",
        new_value="27 December",
        scope="Sinergia company holiday",
        effective_period="" if needs_clarification else "2026",
        reason="Manager correction",
        needs_clarification=needs_clarification,
        clarification_question=(
            "Which year does this correction apply to?" if needs_clarification else ""
        ),
        language="english",
        usage={"inputTokens": 5},
    )


class AcceptedConfidenceEvaluator:
    def evaluate(self, question, result):
        return ConfidenceEvaluation(
            score=9.0,
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


class AnsweringAgent:
    def answer(self, question):
        return AnswerResult(
            status="answered",
            answer="The company holiday starts on 28 December.",
            citations=(Citation("concepts/holiday.md", ("raw/holiday.md",)),),
            usage={},
            pages_read=("concepts/holiday.md",),
        )


class FakeCorrectionInterpreter:
    def __init__(self, value: ManagerActionProposal | None) -> None:
        self.value = value
        self.calls: list[tuple[ManagerActionContext, str, ManagerActionProposal | None]] = []

    @staticmethod
    def looks_like_correction(message):
        return message.casefold().startswith(("no", "correction"))

    looks_like_action = looks_like_correction

    @staticmethod
    def is_confirmation(message):
        return message.casefold().strip() == "confirm"

    @staticmethod
    def is_cancellation(message):
        return message.casefold().strip() == "cancel"

    def interpret(self, context, manager_message, *, draft=None):
        self.calls.append((context, manager_message, draft))
        return self.value


class FakeCorrectionStore:
    def __init__(self) -> None:
        self.calls: list[tuple[ManagerActionProposal, ManagerActionContext]] = []

    def persist_knowledge(self, correction, context):
        self.calls.append((correction, context))
        return "raw/manager-actions/correction123.md"


class CorrectionWorkflowTests(unittest.TestCase):
    def test_interpreter_structures_temporal_correction_and_requests_year(self) -> None:
        bedrock = ScriptedBedrock(correction_turn(needs_clarification=True))
        context = ManagerActionContext(
            question="When does the company holiday start?",
            answer="28 December.",
            citations=(),
        )

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "No, it starts on 27 December.",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.new_value, "27 December")
        self.assertTrue(result.needs_clarification)
        self.assertIn("year", result.clarification_question.casefold())
        self.assertEqual(result.usage, {"inputTokens": 20, "outputTokens": 10})

    def test_correction_commands_are_explicit(self) -> None:
        self.assertTrue(ManagerActionInterpreter.looks_like_action("No, it is 27 December"))
        self.assertTrue(ManagerActionInterpreter.is_confirmation("Yes, confirm"))
        self.assertTrue(ManagerActionInterpreter.is_cancellation("Cancel correction"))
        self.assertFalse(ManagerActionInterpreter.looks_like_action("Tell me more"))

    def test_store_creates_immutable_raw_source_without_llm_generated_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = WikiRepository(Path(temp_dir) / "backend")
            context = ManagerActionContext(
                question="When does the holiday start?",
                answer="28 December.",
                citations=(
                    {
                        "wiki_path": "concepts/holiday.md",
                        "source_paths": ["raw/holiday.md"],
                    },
                ),
            )

            source_path = ManagerActionStore(repository).persist_knowledge(proposal(), context)
            content = repository.read_raw(source_path)

            self.assertTrue(source_path.startswith("raw/manager-actions/"))
            self.assertIn("27 December", content)
            self.assertIn("POC manager", content)
            self.assertNotIn("28 December.", content)
            self.assertNotIn("concepts/holiday.md", content)
            self.assertNotIn("Previous answer", content)
            filename = source_path.rsplit("/", 1)[-1]
            with self.assertRaises(RepositoryError):
                repository.create_manager_correction_source(filename, content)

    def _service(self, temporary_root: str, interpreter, store) -> WikiService:
        settings = BedrockSettings(
            project_root=Path(temporary_root),
            region_name="eu-west-1",
            bedrock_model_id="test-model",
        )
        return WikiService(
            settings,
            agent=AnsweringAgent(),
            confidence_evaluator=AcceptedConfidenceEvaluator(),
            correction_interpreter=interpreter,
            correction_store=store,
        )

    def test_session_proposes_and_applies_confirmed_correction(self) -> None:
        interpreter = FakeCorrectionInterpreter(proposal())
        store = FakeCorrectionStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(temp_dir, interpreter, store)
            initial = service.ask("When does the holiday start?", session_id="manager-1")
            proposed = service.ask(
                "No, it starts on 27 December for 2026.",
                session_id="manager-1",
            )
            update_result = {
                "processed": [
                    {
                        "source_path": "raw/manager-actions/correction123.md",
                        "pages_written": [],
                        "usage": {"inputTokens": 7},
                    }
                ],
                "skipped": [],
                "failed": [],
            }
            with patch.object(
                service, "_update_existing_knowledge", return_value=update_result
            ) as update:
                applied = service.ask("Confirm", session_id="manager-1")

        self.assertEqual(initial["status"], "answered")
        self.assertEqual(proposed["status"], "manager_action_proposed")
        self.assertEqual(proposed["correction"]["corrected_value"], "27 December")
        self.assertEqual(applied["status"], "manager_action_applied")
        self.assertIn("integrated into the Wiki", applied["answer"])
        self.assertEqual(
            applied["correction"]["source_path"],
            "raw/manager-actions/correction123.md",
        )
        self.assertEqual(len(store.calls), 1)
        update.assert_called_once_with(
            "raw/manager-actions/correction123.md",
            proposal=replace(
                proposal(), source_path="raw/manager-actions/correction123.md"
            ),
            context=interpreter.calls[0][0],
        )

    def test_clarification_blocks_confirmation_and_cancel_preserves_wiki(self) -> None:
        interpreter = FakeCorrectionInterpreter(proposal(needs_clarification=True))
        store = FakeCorrectionStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(temp_dir, interpreter, store)
            service.ask("When does the holiday start?", session_id="manager-2")
            proposed = service.ask("No, it is 27 December.", session_id="manager-2")
            still_pending = service.ask("Confirm", session_id="manager-2")
            cancelled = service.ask("Cancel", session_id="manager-2")

        self.assertEqual(proposed["status"], "manager_action_needs_clarification")
        self.assertEqual(still_pending["status"], "manager_action_needs_clarification")
        self.assertEqual(cancelled["status"], "manager_action_cancelled")
        self.assertEqual(store.calls, [])


if __name__ == "__main__":
    unittest.main()
