from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_wiki import (  # noqa: E402
    BedrockJudge,
    JudgeResponse,
    WikiApiClient,
    citation_metrics,
    classify_result,
    collect_wiki_evidence,
    parse_json_object,
    point_coverage,
    main,
    summarize,
    validate_benchmark,
)


class FakeBedrockClient:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "name": "submit_evaluation",
                                "toolUseId": "tool-1",
                                "input": self.evaluation,
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


class FakeValidationException(Exception):
    response = {"Error": {"Code": "ValidationException"}}


class FakeJsonFallbackClient:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls < 3:
            raise FakeValidationException("Tools unavailable")
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": json.dumps(self.evaluation)}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 90, "outputTokens": 18},
        }


def judgment(score=5, groundedness=1.0, unsupported=None):
    return {
        "point_results": [
            {"point_index": 1, "verdict": "covered", "explanation": "Present."},
            {"point_index": 2, "verdict": "partially_covered", "explanation": "Partial."},
        ],
        "correctness_score": score,
        "correctness_explanation": "Evaluation.",
        "missing_information": [],
        "incorrect_claims": [],
        "claim_results": [],
        "groundedness_evaluated": True,
        "groundedness_score": groundedness,
        "unsupported_claims": unsupported or [],
    }


class EvaluationCoreTests(unittest.TestCase):
    def test_real_benchmark_is_valid_and_has_expected_distribution(self):
        path = Path(__file__).resolve().parents[2] / "mateial" / "ground_truth_qa.json"
        if not path.exists():
            self.skipTest("private benchmark fixture is not included in the repository")
        data = validate_benchmark(json.loads(path.read_text(encoding="utf-8-sig")))
        self.assertEqual(len(data["cases"]), 25)
        statuses = [case["expected_status"] for case in data["cases"]]
        self.assertEqual(statuses.count("answered"), 23)
        self.assertEqual(statuses.count("insufficient_knowledge"), 2)

    def test_bedrock_judge_uses_forced_structured_tool(self):
        fake = FakeBedrockClient(judgment())
        judge = BedrockJudge(
            model_id="judge-model",
            region_name="eu-west-1",
            system_prompt="Judge.",
            client=fake,
        )
        result = judge.evaluate({"required_answer_points": ["A", "B"]})
        self.assertEqual(result.evaluation["correctness_score"], 5)
        self.assertEqual(result.evaluation["point_results"][0]["point"], "A")
        self.assertEqual(result.usage["inputTokens"], 100)
        self.assertEqual(result.structured_mode, "forced_tool")
        self.assertEqual(
            fake.requests[0]["toolConfig"]["toolChoice"]["tool"]["name"],
            "submit_evaluation",
        )

    def test_json_text_fallback_parser_accepts_fenced_json(self):
        self.assertEqual(parse_json_object('```json\n{"score": 5}\n```'), {"score": 5})

    def test_bedrock_judge_falls_back_to_json_when_tools_are_unsupported(self):
        fake = FakeJsonFallbackClient(judgment())
        judge = BedrockJudge(
            model_id="judge-model",
            region_name="eu-west-1",
            system_prompt="Judge.",
            client=fake,
        )
        result = judge.evaluate({"required_answer_points": ["A", "B"]})
        self.assertEqual(result.structured_mode, "json_text")
        self.assertEqual(result.evaluation["correctness_score"], 5)
        self.assertEqual(fake.calls, 3)

    def test_citation_metrics_normalize_backend_prefix(self):
        case = {"sources": [{"source_path": "raw/Folder/Doc.pdf"}]}
        response = {
            "citations": [
                {
                    "wiki_path": "sources/doc.md",
                    "source_paths": ["WIKI/backend/raw/Folder/Doc.pdf", "raw/other.txt"],
                }
            ]
        }
        metrics = citation_metrics(case, response)
        self.assertEqual(metrics["expected_source_recall"], 1.0)
        self.assertEqual(metrics["expected_source_precision"], 0.5)

    def test_evidence_loader_rejects_traversal_and_bounds_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = root / "sources" / "page.md"
            page.parent.mkdir()
            page.write_text("abcdefghij", encoding="utf-8")
            pages, metadata = collect_wiki_evidence(
                root,
                ["sources/page.md", "../secret.txt"],
                max_chars_per_page=4,
                max_total_chars=10,
            )
            self.assertEqual(pages[0]["content"], "abcd")
            self.assertTrue(metadata[0]["truncated"])
            self.assertFalse(metadata[1]["available"])

    def test_point_coverage_uses_half_credit(self):
        points = [
            {"verdict": "covered"},
            {"verdict": "partially_covered"},
            {"verdict": "missing"},
            {"verdict": "contradicted"},
        ]
        self.assertEqual(point_coverage(points), 0.375)

    def test_failure_classification_separates_lookup_and_generation(self):
        case = {"expected_status": "answered"}
        response = {"status": "answered", "citations": [{"wiki_path": "x"}]}
        low = judgment(score=2)
        primary, flags = classify_result(
            case, response, low, {"expected_source_recall": 0.0}
        )
        self.assertEqual(primary, "INCORRECT")
        self.assertIn("WIKI_LOOKUP_FAILURE", flags)
        primary, flags = classify_result(
            case, response, low, {"expected_source_recall": 1.0}
        )
        self.assertIn("ANSWER_GENERATION_FAILURE", flags)

    def test_summary_aggregates_scores_latency_usage_and_cost(self):
        result = {
            "case_id": "qa-001",
            "expected_status": "answered",
            "primary_outcome": "CORRECT",
            "diagnostic_flags": [],
            "required_point_coverage": 0.75,
            "citation_metrics": {"expected_source_recall": 1.0},
            "client_latency_ms": 120.0,
            "chatbot": {
                "status": "answered",
                "latency_ms": 100.0,
                "usage": {"inputTokens": 1000, "outputTokens": 200},
            },
            "judgment": {
                "correctness_score": 4,
                "groundedness_evaluated": True,
                "groundedness_score": 0.8,
            },
            "judge": {"usage": {"inputTokens": 2000, "outputTokens": 500}},
        }
        value = summarize(
            [result],
            {
                "chatbot_input_per_million_usd": 1.0,
                "chatbot_output_per_million_usd": 2.0,
                "judge_input_per_million_usd": 3.0,
                "judge_output_per_million_usd": 4.0,
            },
        )
        self.assertEqual(value["correctness"]["average_score_1_to_5"], 4.0)
        self.assertEqual(value["latency_ms"]["server_p95"], 100.0)
        self.assertEqual(value["usage"]["judge"]["inputTokens"], 2000)
        self.assertAlmostEqual(value["estimated_cost_usd"]["total"], 0.0094)

    def test_main_writes_complete_run_artifacts_without_live_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wiki = root / "wiki"
            (wiki / "sources").mkdir(parents=True)
            (wiki / "sources" / "page.md").write_text("Evidence.", encoding="utf-8")
            dataset = root / "benchmark.json"
            dataset.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "cases": [
                            {
                                "id": "qa-test",
                                "question": "Question?",
                                "expected_status": "answered",
                                "ground_truth_answer": "Answer.",
                                "required_answer_points": ["A", "B"],
                                "question_type": "fact",
                                "difficulty": "easy",
                                "sources": [{"source_path": "raw/doc.txt"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = root / "config.json"
            output = root / "results"
            config.write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "wiki_root": str(wiki),
                        "output_dir": str(output),
                        "judge": {
                            "model_id": "independent-judge",
                            "prompt": str(Path(__file__).resolve().parents[1] / "judge_prompt.md"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            chatbot = {
                "approach": "wiki",
                "status": "answered",
                "answer": "Answer.",
                "citations": [
                    {"wiki_path": "sources/page.md", "source_paths": ["raw/doc.txt"]}
                ],
                "usage": {"inputTokens": 10, "outputTokens": 2},
                "latency_ms": 15.0,
                "model_id": "chatbot-model",
                "debug": {"pages_read": ["sources/page.md"], "search_queries": []},
            }
            judged = JudgeResponse(
                evaluation={
                    **judgment(),
                    "point_results": [
                        {"point_index": 1, "point": "A", "verdict": "covered", "explanation": "Yes."},
                        {"point_index": 2, "point": "B", "verdict": "covered", "explanation": "Yes."},
                    ],
                },
                usage={"inputTokens": 20, "outputTokens": 5},
                latency_ms=25.0,
                stop_reason="tool_use",
                structured_mode="forced_tool",
            )
            with (
                patch.object(WikiApiClient, "health", return_value={"bedrock_configured": True, "model_id": "chatbot-model"}),
                patch.object(WikiApiClient, "lint", return_value={"valid": True, "pages_checked": 1, "graph": {}}),
                patch.object(WikiApiClient, "chat", return_value=(chatbot, 20.0)),
                patch.object(BedrockJudge, "evaluate", return_value=judged),
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--config", str(config)]), 0)

            run_dirs = list(output.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "results.jsonl").is_file())
            self.assertTrue((run_dir / "summary.csv").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["correctness"]["average_score_1_to_5"], 5.0)


if __name__ == "__main__":
    unittest.main()
