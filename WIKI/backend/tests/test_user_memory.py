from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.agent import AnswerResult
from backend.app.bedrock import BedrockError, ConverseTurn
from backend.app.config import BedrockSettings
from backend.app.preference_interpreter import (
    PreferenceDecision,
    PreferenceInterpreter,
)
from backend.app.repository import WikiRepository
from backend.app.service import WikiService
from backend.app.user_memory import (
    UserMemoryError,
    UserMemoryStore,
)


class ContextAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def answer(
        self,
        question,
        *,
        conversation_history=(),
        user_preferences=(),
    ):
        self.calls.append(
            {
                "question": question,
                "conversation_history": tuple(conversation_history),
                "user_preferences": tuple(user_preferences),
            }
        )
        return AnswerResult(
            status="insufficient_knowledge",
            answer=f"No Wiki evidence for: {question}",
            citations=(),
            usage={},
        )


class UnavailableConfidence:
    def evaluate(self, question, result):
        raise RuntimeError("not needed in this test")


def preference_decision(
    operation: str = "none",
    *,
    intent_kind=None,
    add=(),
    remove=(),
    remaining_question="",
    explicit=True,
    requires_clarification=False,
    clarification_question="",
    language="english",
) -> PreferenceDecision:
    return PreferenceDecision(
        intent_kind=intent_kind
        or {
            "none": "no_preference",
            "temporary": "temporary_behavior",
            "add": "persistent_behavior",
            "replace": "persistent_behavior",
            "remove": "memory_deletion",
            "clear": "memory_deletion",
        }[operation],
        operation=operation,
        preferences_to_add=tuple(add),
        preferences_to_remove=tuple(remove),
        remaining_question=remaining_question,
        explicit=explicit,
        requires_clarification=requires_clarification,
        clarification_question=clarification_question,
        confidence=0.99,
        language=language,
        explanation="test decision",
        usage={"inputTokens": 5, "outputTokens": 3},
        attempts=1,
    )


class ScriptedPreferenceInterpreter:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def interpret(self, message, *, current_preferences, conversation_history=()):
        self.calls.append(
            {
                "message": message,
                "current_preferences": tuple(current_preferences),
                "conversation_history": tuple(conversation_history),
            }
        )
        return self.decisions.pop(0)


class ScriptedBedrock:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def preference_turn(payload) -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "preference-1",
                        "name": "submit_preference_decision",
                        "input": payload,
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 11, "outputTokens": 7},
        metrics={},
    )


class UserMemoryStoreTests(unittest.TestCase):
    def test_profile_and_exact_session_history_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "user_data"
            store = UserMemoryStore(root)
            profile = store.save_profile(
                "user1",
                ["Always answer me in Italian.", "Keep answers concise."],
            )

            store.append_exchange(
                "user1",
                "session-1",
                "What is the policy?",
                {
                    "status": "answered",
                    "answer": "The policy is documented.",
                    "citations": [
                        {
                            "wiki_path": "sources/policy.md",
                            "source_paths": ["raw/policy.pdf"],
                        }
                    ],
                },
            )

            self.assertEqual(profile.preferences[0], "Always answer me in Italian.")
            self.assertTrue((root / "user1" / "preferences.json").is_file())
            self.assertIn(
                "Always answer me in Italian.",
                (root / "user1" / "profile.md").read_text(encoding="utf-8"),
            )
            session_path = root / "user1" / "sessions" / "session-1.jsonl"
            events = [
                json.loads(line)
                for line in session_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["role"] for event in events], ["user", "assistant"])
            self.assertEqual(events[0]["content"], "What is the policy?")
            self.assertEqual(events[1]["content"], "The policy is documented.")
            self.assertEqual(len(store.read_session_context("user1", "session-1")), 2)

    def test_clear_session_deletes_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = UserMemoryStore(Path(temp_dir) / "user_data")
            store.save_profile("user1", ["Answer in Italian."])
            store.append_exchange(
                "user1",
                "session-1",
                "Question",
                {"status": "answered", "answer": "Answer", "citations": []},
            )

            self.assertTrue(store.clear_session("user1", "session-1"))
            self.assertFalse(store.clear_session("user1", "session-1"))
            self.assertEqual(
                store.get_profile("user1").preferences,
                ("Answer in Italian.",),
            )

    def test_unsafe_identifiers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = UserMemoryStore(Path(temp_dir) / "user_data")
            with self.assertRaises(UserMemoryError):
                store.get_profile("../other-user")
            with self.assertRaises(UserMemoryError):
                store.clear_session("user1", "../../other-session")

    def test_preference_changes_are_atomic_and_preserve_unrelated_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = UserMemoryStore(Path(temp_dir) / "user_data")
            store.save_profile(
                "user1",
                ["Always answer in Italian.", "Keep answers concise."],
            )
            profile = store.apply_preference_changes(
                "user1",
                preferences_to_remove=["Always answer in Italian."],
                preferences_to_add=["Always answer in English."],
            )
            self.assertEqual(
                profile.preferences,
                ("Keep answers concise.", "Always answer in English."),
            )
            with self.assertRaises(UserMemoryError):
                store.apply_preference_changes(
                    "user1",
                    preferences_to_remove=["Unknown preference"],
                )


class PreferenceInterpreterTests(unittest.TestCase):
    def test_structured_replace_uses_exact_existing_preferences(self) -> None:
        current = (
            "always answer me in italian",
            "always after you answer me in italian then write same answer in english",
        )
        bedrock = ScriptedBedrock(
            [
                preference_turn(
                    {
                        "operation": "replace",
                        "intent_kind": "persistent_behavior",
                        "preferences_to_add": ["Always answer me in English."],
                        "preferences_to_remove": list(current),
                        "remaining_question": "",
                        "explicit": True,
                        "requires_clarification": False,
                        "clarification_question": "",
                        "confidence": 0.99,
                        "language": "english",
                        "explanation": "The new durable instruction conflicts with both.",
                    }
                )
            ]
        )
        decision = PreferenceInterpreter(bedrock).interpret(
            "never answer me in italian",
            current_preferences=current,
        )
        self.assertEqual(decision.operation, "replace")
        self.assertEqual(decision.preferences_to_remove, current)
        self.assertEqual(
            decision.preferences_to_add,
            ("Always answer me in English.",),
        )
        self.assertTrue(decision.is_preference_only)
        self.assertEqual(decision.usage["inputTokens"], 11)

    def test_transient_bedrock_failure_is_retried_once(self) -> None:
        bedrock = ScriptedBedrock(
            [
                BedrockError("temporary outage"),
                preference_turn(
                    {
                        "operation": "none",
                        "intent_kind": "no_preference",
                        "preferences_to_add": [],
                        "preferences_to_remove": [],
                        "remaining_question": "When is the meeting?",
                        "explicit": False,
                        "requires_clarification": False,
                        "clarification_question": "",
                        "confidence": 0.99,
                        "language": "english",
                        "explanation": "No preference request.",
                    }
                ),
            ]
        )
        decision = PreferenceInterpreter(bedrock).interpret(
            "When is the meeting?",
            current_preferences=(),
        )
        self.assertEqual(decision.operation, "none")
        self.assertEqual(decision.attempts, 2)
        self.assertEqual(len(bedrock.calls), 2)

    def test_behavioral_prohibition_cannot_be_accepted_as_memory_deletion(self) -> None:
        current = ("Always answer me in Italian.",)
        invalid = preference_turn(
            {
                "operation": "remove",
                "intent_kind": "persistent_behavior",
                "preferences_to_add": [],
                "preferences_to_remove": list(current),
                "remaining_question": "",
                "explicit": True,
                "requires_clarification": False,
                "clarification_question": "",
                "confidence": 0.99,
                "language": "english",
                "explanation": "Incorrectly treated a behavioral rule as deletion.",
            }
        )
        repaired = preference_turn(
            {
                "operation": "replace",
                "intent_kind": "persistent_behavior",
                "preferences_to_add": ["Never answer me in Italian."],
                "preferences_to_remove": list(current),
                "remaining_question": "",
                "explicit": True,
                "requires_clarification": False,
                "clarification_question": "",
                "confidence": 0.99,
                "language": "english",
                "explanation": "Retain the durable behavioral prohibition.",
            }
        )
        bedrock = ScriptedBedrock([invalid, repaired])
        decision = PreferenceInterpreter(bedrock).interpret(
            "never answer me in italian",
            current_preferences=current,
        )
        self.assertEqual(decision.operation, "replace")
        self.assertEqual(
            decision.preferences_to_add,
            ("Never answer me in Italian.",),
        )
        self.assertEqual(decision.attempts, 2)

    def test_low_confidence_persistent_change_requires_clarification(self) -> None:
        bedrock = ScriptedBedrock(
            [
                preference_turn(
                    {
                        "operation": "add",
                        "intent_kind": "persistent_behavior",
                        "preferences_to_add": ["Use a formal tone."],
                        "preferences_to_remove": [],
                        "remaining_question": "",
                        "explicit": True,
                        "requires_clarification": False,
                        "clarification_question": "",
                        "confidence": 0.6,
                        "language": "english",
                        "explanation": "Duration is uncertain.",
                    }
                )
            ]
        )
        decision = PreferenceInterpreter(bedrock).interpret(
            "Maybe use a formal tone",
            current_preferences=(),
        )
        self.assertTrue(decision.requires_clarification)
        self.assertTrue(decision.clarification_question)


class UserMemoryServiceTests(unittest.TestCase):
    def test_preferences_survive_new_chat_while_session_context_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            backend_root.mkdir(parents=True)
            (backend_root / "AGENTS.md").write_text(
                "Maintain a cited wiki.",
                encoding="utf-8",
            )
            settings = BedrockSettings(
                project_root=project_root,
                region_name="eu-west-1",
                bedrock_model_id="test-model",
            )
            store = UserMemoryStore(backend_root / "user_data")
            agent = ContextAgent()
            preference_interpreter = ScriptedPreferenceInterpreter(
                [
                    preference_decision(
                        "add",
                        add=("Always answer me in Italian.",),
                    ),
                    preference_decision(
                        remaining_question="First factual question",
                    ),
                    preference_decision(
                        remaining_question="What about the second point?",
                    ),
                    preference_decision(
                        remaining_question="Question in the new chat",
                    ),
                ]
            )
            service = WikiService(
                settings,
                repository=WikiRepository(backend_root),
                agent=agent,
                confidence_evaluator=UnavailableConfidence(),
                user_memory_store=store,
                preference_interpreter=preference_interpreter,
            )

            empty_profile = service.get_user_profile("user1")
            self.assertEqual(empty_profile["preferences"], [])
            self.assertTrue(
                (backend_root / "user_data" / "user1" / "profile.md").is_file()
            )

            preference = service.ask(
                "Always answer me in Italian.",
                user_id="user1",
                session_id="session-1",
            )
            first = service.ask(
                "First factual question",
                user_id="user1",
                session_id="session-1",
            )
            second = service.ask(
                "What about the second point?",
                user_id="user1",
                session_id="session-1",
            )

            self.assertEqual(preference["status"], "preference_saved")
            self.assertTrue(first["debug"]["history_saved"])
            self.assertEqual(agent.calls[0]["user_preferences"], (
                "Always answer me in Italian.",
            ))
            self.assertEqual(len(agent.calls[1]["conversation_history"]), 2)
            self.assertEqual(
                agent.calls[1]["conversation_history"][0]["content"],
                "First factual question",
            )
            self.assertEqual(second["debug"]["history_messages_used"], 2)

            reset = service.reset_chat("user1", "session-1")
            self.assertTrue(reset["history_deleted"])
            self.assertEqual(
                service.get_user_profile("user1")["preferences"],
                ["Always answer me in Italian."],
            )
            service.ask(
                "Question in the new chat",
                user_id="user1",
                session_id="session-2",
            )
            self.assertEqual(agent.calls[2]["conversation_history"], ())
            self.assertEqual(agent.calls[2]["user_preferences"], (
                "Always answer me in Italian.",
            ))

    def test_conflicting_preference_is_replaced_without_wiki_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            backend_root.mkdir(parents=True)
            (backend_root / "AGENTS.md").write_text("Maintain a cited wiki.", encoding="utf-8")
            store = UserMemoryStore(backend_root / "user_data")
            old_preferences = (
                "always answer me in italian",
                "always after you answer me in italian then write same answer in english",
            )
            store.save_profile("user1", old_preferences)
            agent = ContextAgent()
            interpreter = ScriptedPreferenceInterpreter(
                [
                    preference_decision(
                        "replace",
                        add=("Always answer me in English.",),
                        remove=old_preferences,
                    )
                ]
            )
            service = WikiService(
                BedrockSettings(
                    project_root=project_root,
                    region_name="eu-west-1",
                    bedrock_model_id="test-model",
                ),
                repository=WikiRepository(backend_root),
                agent=agent,
                confidence_evaluator=UnavailableConfidence(),
                user_memory_store=store,
                preference_interpreter=interpreter,
            )
            response = service.ask(
                "never answer me in italian",
                user_id="user1",
                session_id="session-1",
            )
            self.assertEqual(response["status"], "preferences_updated")
            self.assertEqual(response["citations"], [])
            self.assertTrue(response["preference_changed"])
            self.assertEqual(agent.calls, [])
            self.assertEqual(
                store.get_profile("user1").preferences,
                ("Always answer me in English.",),
            )
            self.assertEqual(store.read_session_context("user1", "session-1"), ())

    def test_mixed_preference_update_answers_remaining_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_root = project_root / "backend"
            backend_root.mkdir(parents=True)
            (backend_root / "AGENTS.md").write_text("Maintain a cited wiki.", encoding="utf-8")
            store = UserMemoryStore(backend_root / "user_data")
            agent = ContextAgent()
            interpreter = ScriptedPreferenceInterpreter(
                [
                    preference_decision(
                        "add",
                        add=("Always use bullet points.",),
                        remaining_question="When is the meeting?",
                    )
                ]
            )
            service = WikiService(
                BedrockSettings(
                    project_root=project_root,
                    region_name="eu-west-1",
                    bedrock_model_id="test-model",
                ),
                repository=WikiRepository(backend_root),
                agent=agent,
                confidence_evaluator=UnavailableConfidence(),
                user_memory_store=store,
                preference_interpreter=interpreter,
            )
            response = service.ask(
                "Always use bullet points. When is the meeting?",
                user_id="user1",
                session_id="session-1",
            )
            self.assertEqual(agent.calls[0]["question"], "When is the meeting?")
            self.assertEqual(
                agent.calls[0]["user_preferences"],
                ("Always use bullet points.",),
            )
            self.assertTrue(response["preference_changed"])
            self.assertEqual(response["preference_operation"], "add")


if __name__ == "__main__":
    unittest.main()
