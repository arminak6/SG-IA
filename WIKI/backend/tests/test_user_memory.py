from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.agent import AnswerResult
from backend.app.config import BedrockSettings
from backend.app.repository import WikiRepository
from backend.app.service import WikiService
from backend.app.user_memory import (
    UserMemoryError,
    UserMemoryStore,
    detect_preference_instruction,
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

    def test_only_explicit_durable_phrasing_is_detected(self) -> None:
        self.assertEqual(
            detect_preference_instruction("Always answer me in Italian.").action,
            "add",
        )
        self.assertEqual(
            detect_preference_instruction("/remember Keep answers concise.").value,
            "Keep answers concise.",
        )
        self.assertEqual(
            detect_preference_instruction("Forget my preferences.").action,
            "clear",
        )
        self.assertIsNone(
            detect_preference_instruction("Can you answer this one in Italian?")
        )


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
            service = WikiService(
                settings,
                repository=WikiRepository(backend_root),
                agent=agent,
                confidence_evaluator=UnavailableConfidence(),
                user_memory_store=store,
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


if __name__ == "__main__":
    unittest.main()
