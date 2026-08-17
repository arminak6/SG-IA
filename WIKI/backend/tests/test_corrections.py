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
                            "new_value": "It starts on 27 December.",
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


def unstructured_turn() -> ConverseTurn:
    return ConverseTurn(
        message={"role": "assistant", "content": [{"text": "Please clarify."}]},
        stop_reason="end_turn",
        usage={"inputTokens": 5, "outputTokens": 2},
        metrics={},
    )


def dropped_recurring_update_turn() -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "correction-recurring",
                        "name": "submit_manager_action",
                        "input": {
                            "is_manager_action": True,
                            "action_type": "update_knowledge",
                            "subject": "Sinergia mid-spring meeting",
                            "previous_value": "The meeting was on 17 February.",
                            "new_value": "",
                            "scope": "entities/sinergia-mid-spring-meeting.md",
                            "effective_period": "",
                            "reason": "Manager correction",
                            "needs_clarification": True,
                            "clarification_question": (
                                "Could you provide the effective period for the new date?"
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


def mid_summer_merged_turn() -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "correction-mid-summer",
                        "name": "submit_manager_action",
                        "input": {
                            "is_manager_action": True,
                            "action_type": "fix_answer",
                            "subject": "Sinergia mid-summer meeting",
                            "previous_value": "",
                            "new_value": (
                                "The annual Sinergia mid-summer meeting is expected on "
                                "13 July with 99% certainty. Sinergia will send a reminder "
                                "email one week beforehand to confirm the date will be observed."
                            ),
                            "scope": "",
                            "effective_period": "",
                            "reason": "The manager qualified the date and added confirmation details.",
                            "needs_clarification": True,
                            "clarification_question": (
                                "Please specify whether this fixes the previous answer, updates "
                                "existing knowledge, or adds new knowledge."
                            ),
                            "language": "english",
                        },
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 30, "outputTokens": 20},
        metrics={},
    )


def merge_review_turn() -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "merge-review",
                        "name": "review_manager_merge",
                        "input": {
                            "valid": False,
                            "corrected_value": (
                                "The annual Sinergia mid-summer meeting is expected on "
                                "13 July with 99% certainty. Sinergia will email everyone "
                                "one week beforehand to confirm the date."
                            ),
                            "unsupported_additions": [
                                "The email was characterized as a reminder.",
                                "The date was characterized as being observed.",
                            ],
                            "explanation": "Those characterizations are not stated in the inputs.",
                        },
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 25, "outputTokens": 15},
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
        return message.casefold().startswith(("/fix", "/update", "/add"))

    looks_like_action = looks_like_correction

    @staticmethod
    def is_confirmation(message):
        return message.casefold().strip().lstrip("/") in {
            "confirm",
            "approve",
            "confermo",
            "approva",
        }

    @staticmethod
    def is_cancellation(message):
        return message.casefold().strip().lstrip("/") == "cancel"

    @staticmethod
    def is_control_command(message):
        return message.casefold().strip() in {"/confirm", "/cancel"}

    @staticmethod
    def looks_like_unmarked_action(message):
        return False

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
    def test_update_button_owns_action_and_cited_scope_without_reclassification(self) -> None:
        manager_text = (
            "13 July is 99% expected, but Sinergia will email everybody one week "
            "in advance to confirm the date."
        )
        context = ManagerActionContext(
            question="When is the Sinergia mid-summer meeting?",
            answer="The Sinergia mid-summer meeting is held annually on 13 July.",
            citations=(
                {
                    "wiki_path": "entities/sinergia-mid-summer-meeting.md",
                    "source_paths": ["raw/manager-knowledge/sinergia-meeting.md"],
                },
                {
                    "wiki_path": "sources/sinergia-meeting.md",
                    "source_paths": ["raw/manager-knowledge/sinergia-meeting.md"],
                },
            ),
            maintained_knowledge=(
                {
                    "source_path": "raw/manager-knowledge/sinergia-meeting.md",
                    "current_value": (
                        "The meeting is held annually on 13 July. Sinergia will email "
                        "everyone one week beforehand to confirm the date."
                    ),
                },
            ),
        )

        result = ManagerActionInterpreter(
            ScriptedBedrock(mid_summer_merged_turn(), merge_review_turn())
        ).interpret(context, f"/update {manager_text}")

        self.assertIsNotNone(result)
        self.assertEqual(result.action_type, "update_knowledge")
        self.assertEqual(result.previous_value, context.answer)
        self.assertEqual(
            result.scope,
            "entities/sinergia-mid-summer-meeting.md, sources/sinergia-meeting.md",
        )
        self.assertIn("99% certainty", result.new_value)
        self.assertIn("confirm the date", result.new_value)
        self.assertNotIn("6 July", result.new_value)
        self.assertNotIn("reminder", result.new_value)
        self.assertNotIn("observed", result.new_value)
        self.assertEqual(len(result.merge_warnings), 2)
        self.assertEqual(result.manager_input, manager_text)
        self.assertEqual(result.effective_period, "annual")
        self.assertFalse(result.needs_clarification)
        self.assertTrue(result.ready_for_confirmation)

    def test_recurring_update_preserves_manager_value_without_clarification(self) -> None:
        bedrock = ScriptedBedrock(dropped_recurring_update_turn())
        context = ManagerActionContext(
            question="When is the mid-spring meeting?",
            answer="The meeting was on 17 February.",
            citations=(),
        )

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "/update The meeting is on 26 February every year.",
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.new_value,
            "The meeting is on 26 February every year.",
        )
        self.assertEqual(result.effective_period, "every year")
        self.assertFalse(result.needs_clarification)
        self.assertTrue(result.ready_for_confirmation)

    def test_interpreter_structures_temporal_correction_and_requests_year(self) -> None:
        bedrock = ScriptedBedrock(correction_turn(needs_clarification=True))
        context = ManagerActionContext(
            question="When does the company holiday start?",
            answer="28 December.",
            citations=(),
        )

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "/update It starts on 27 December.",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.new_value, "It starts on 27 December.")
        self.assertTrue(result.needs_clarification)
        self.assertIn("year", result.clarification_question.casefold())
        self.assertEqual(result.usage, {"inputTokens": 20, "outputTokens": 10})

    def test_correction_commands_are_explicit(self) -> None:
        self.assertTrue(ManagerActionInterpreter.looks_like_action("/update It is 27 December"))
        self.assertTrue(ManagerActionInterpreter.looks_like_action("/fix The answer is wrong"))
        self.assertTrue(ManagerActionInterpreter.looks_like_action("/add New policy"))
        self.assertFalse(ManagerActionInterpreter.looks_like_action("No, it is 27 December"))
        self.assertTrue(ManagerActionInterpreter.is_confirmation("/confirm"))
        self.assertTrue(ManagerActionInterpreter.is_confirmation("approve"))
        self.assertTrue(ManagerActionInterpreter.is_confirmation("/approva"))
        self.assertFalse(ManagerActionInterpreter.is_confirmation("approved"))
        self.assertTrue(ManagerActionInterpreter.is_cancellation("/cancel"))
        self.assertFalse(ManagerActionInterpreter.is_confirmation("yes"))
        self.assertFalse(ManagerActionInterpreter.looks_like_action("Tell me more"))

    def test_interpreter_does_not_call_model_without_explicit_command(self) -> None:
        bedrock = ScriptedBedrock(correction_turn())
        context = ManagerActionContext(
            question="When does the company holiday start?",
            answer="28 December.",
            citations=(),
        )

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "No, it starts on 27 December.",
        )

        self.assertIsNone(result)
        self.assertEqual(bedrock.calls, [])

    def test_explicit_command_determines_action_type(self) -> None:
        bedrock = ScriptedBedrock(correction_turn())
        context = ManagerActionContext(question="", answer="", citations=())

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "/add The company holiday starts on 27 December.",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.action_type, "add_knowledge")

    def test_unstructured_action_result_becomes_short_clarification(self) -> None:
        bedrock = ScriptedBedrock(unstructured_turn(), unstructured_turn())
        context = ManagerActionContext(
            question="When is the event?",
            answer="Its date is not fixed.",
            citations=(),
        )

        result = ManagerActionInterpreter(bedrock).interpret(
            context,
            "/update 24 February or 29 February",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.action_type, "update_knowledge")
        self.assertTrue(result.needs_clarification)
        self.assertIn("single new value", result.clarification_question)

    def test_repeated_same_command_is_removed_before_model_call(self) -> None:
        bedrock = ScriptedBedrock(correction_turn())
        context = ManagerActionContext(
            question="When is the event?",
            answer="Its date is not fixed.",
            citations=(),
        )

        ManagerActionInterpreter(bedrock).interpret(
            context,
            "/update /update It is now 24 February 2027.",
        )

        prompt = bedrock.calls[0]["messages"][0]["content"][0]["text"]
        self.assertIn('"manager_message": "It is now 24 February 2027."', prompt)

    def test_orphan_confirmation_does_not_fall_through_to_qa(self) -> None:
        interpreter = FakeCorrectionInterpreter(proposal())
        store = FakeCorrectionStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(temp_dir, interpreter, store)
            result = service.ask("/confirm", session_id="missing-draft")

        self.assertEqual(result["status"], "manager_action_not_pending")
        self.assertIn("no pending manager action", result["answer"].casefold())
        self.assertEqual(interpreter.calls, [])

    def test_approve_without_pending_draft_remains_qa(self) -> None:
        interpreter = FakeCorrectionInterpreter(proposal())
        store = FakeCorrectionStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(temp_dir, interpreter, store)
            result = service.ask("approve", session_id="ordinary-question")

        self.assertEqual(result["status"], "answered")
        self.assertEqual(interpreter.calls, [])
        self.assertEqual(store.calls, [])

    def test_store_creates_one_stable_source_without_llm_generated_context(self) -> None:
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

            store = ManagerActionStore(repository)
            write = store.persist_knowledge(proposal(), context)
            content = repository.read_raw(write.source_path)

            self.assertEqual(
                write.source_path,
                "raw/manager-knowledge/company-holiday-start-date.md",
            )
            self.assertIn("27 December", content)
            self.assertIn("POC manager", content)
            self.assertNotIn("28 December.", content)
            self.assertNotIn("concepts/holiday.md", content)
            self.assertNotIn("Previous answer", content)
            second = store.persist_knowledge(
                replace(proposal(), new_value="26 December"), context
            )
            self.assertEqual(second.source_path, write.source_path)
            self.assertEqual(len(repository.list_raw_documents()), 1)
            self.assertIn("26 December", repository.read_raw(second.source_path))

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
                "/update It starts on 27 December for 2026.",
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
                applied = service.ask("approve", session_id="manager-1")

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
            proposed = service.ask("/update It is 27 December.", session_id="manager-2")
            still_pending = service.ask("approve", session_id="manager-2")
            cancelled = service.ask("/cancel", session_id="manager-2")

        self.assertEqual(proposed["status"], "manager_action_needs_clarification")
        self.assertEqual(still_pending["status"], "manager_action_needs_clarification")
        self.assertIn("more information required", still_pending["answer"].casefold())
        self.assertNotIn("approval required", still_pending["answer"].casefold())
        self.assertEqual(cancelled["status"], "manager_action_cancelled")
        self.assertEqual(store.calls, [])


if __name__ == "__main__":
    unittest.main()
