from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.agent import AnswerResult, Citation
from backend.app.bedrock import ConverseTurn
from backend.app.confidence import ConfidenceEvaluator
from backend.app.repository import WikiRepository


def wiki_page() -> str:
    return """---
title: Team naming
page_type: concept
updated: 2026-08-11
sources:
  - raw/assumptions.txt
---
# Team naming

The LT Team is called Digitalization in external communication.

## Sources

- raw/assumptions.txt
"""


def confidence_turn(
    *,
    unsupported: bool = False,
    conflict: bool = False,
) -> ConverseTurn:
    return ConverseTurn(
        message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "confidence-1",
                        "name": "submit_confidence_evaluation",
                        "input": {
                            "claim_support": 0.9,
                            "question_coverage": 1.0,
                            "source_consistency": 1.0,
                            "evidence_quality": 0.9,
                            "abstention_appropriateness": 0.0,
                            "has_unsupported_material_claim": unsupported,
                            "has_unexplained_conflict": conflict,
                        },
                    }
                }
            ],
        },
        stop_reason="tool_use",
        usage={"inputTokens": 30, "outputTokens": 10},
        metrics={},
    )


class ScriptedBedrock:
    def __init__(self, *turns: ConverseTurn) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("Unexpected Bedrock call")
        return self.turns.pop(0)


class ConfidenceEvaluatorTests(unittest.TestCase):
    def _fixture(self, bedrock: ScriptedBedrock):
        temporary = tempfile.TemporaryDirectory()
        backend_root = Path(temporary.name) / "backend"
        raw_root = backend_root / "raw"
        raw_root.mkdir(parents=True)
        (raw_root / "assumptions.txt").write_text("Naming source", encoding="utf-8")
        repository = WikiRepository(backend_root)
        repository.write_wiki_pages({"concepts/team-naming.md": wiki_page()})
        result = AnswerResult(
            status="answered",
            answer="The LT Team is called Digitalization externally.",
            citations=(
                Citation(
                    wiki_path="concepts/team-naming.md",
                    source_paths=("raw/assumptions.txt",),
                ),
            ),
            usage={},
            pages_read=("concepts/team-naming.md",),
            search_queries=("team names",),
            search_modes=("hybrid_section",),
            retrieval_diagnostics=(
                {
                    "query": "team names",
                    "mode": "hybrid_section",
                    "candidates": [
                        {
                            "path": "concepts/team-naming.md",
                            "lexical_rank": 1,
                            "semantic_rank": 1,
                        }
                    ],
                },
            ),
        )
        return temporary, repository, result

    def test_answer_score_combines_verifier_and_retrieval_signals(self) -> None:
        bedrock = ScriptedBedrock(confidence_turn())
        temporary, repository, result = self._fixture(bedrock)
        self.addCleanup(temporary.cleanup)

        evaluation = ConfidenceEvaluator(repository, bedrock).evaluate(
            "Which external name is used?", result
        )

        self.assertEqual(evaluation.score, 9.5)
        self.assertEqual(evaluation.usage, {"inputTokens": 30, "outputTokens": 10})
        supplied_evidence = bedrock.calls[0]["messages"][0]["content"][0]["text"]
        self.assertIn("<wiki_page path=\"concepts/team-naming.md\">", supplied_evidence)
        self.assertIn("raw/assumptions.txt", supplied_evidence)

    def test_unsupported_material_claim_caps_score_at_five(self) -> None:
        bedrock = ScriptedBedrock(confidence_turn(unsupported=True))
        temporary, repository, result = self._fixture(bedrock)
        self.addCleanup(temporary.cleanup)

        evaluation = ConfidenceEvaluator(repository, bedrock).evaluate(
            "Which external name is used?", result
        )

        self.assertEqual(evaluation.score, 5.0)

    def test_empty_wiki_abstention_is_confident_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = WikiRepository(Path(temp_dir) / "backend")
            bedrock = ScriptedBedrock()
            result = AnswerResult(
                status="insufficient_knowledge",
                answer="The Wiki has no knowledge pages.",
                citations=(),
                usage={},
            )

            evaluation = ConfidenceEvaluator(repository, bedrock).evaluate(
                "Unknown question", result
            )

        self.assertEqual(evaluation.score, 10.0)
        self.assertEqual(bedrock.calls, [])

    def test_abstention_score_measures_confidence_in_refusal(self) -> None:
        turn = confidence_turn()
        turn.message["content"][0]["toolUse"]["input"][
            "abstention_appropriateness"
        ] = 0.9
        bedrock = ScriptedBedrock(turn)
        temporary, repository, answered = self._fixture(bedrock)
        self.addCleanup(temporary.cleanup)
        result = AnswerResult(
            status="insufficient_knowledge",
            answer="The Wiki does not contain enough evidence.",
            citations=(),
            usage={},
            pages_read=answered.pages_read,
            search_queries=answered.search_queries,
            search_modes=answered.search_modes,
            retrieval_diagnostics=answered.retrieval_diagnostics,
        )

        evaluation = ConfidenceEvaluator(repository, bedrock).evaluate(
            "An unsupported question", result
        )

        self.assertEqual(evaluation.score, 9.3)


if __name__ == "__main__":
    unittest.main()
