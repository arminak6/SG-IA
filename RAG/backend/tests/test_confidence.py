from __future__ import annotations

from app.confidence import BedrockRagConfidenceEvaluator
from app.generation import GeneratedAnswer
from app.models import SearchHit


class FakeBedrockClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def response(inputs=None, *, usage=None):
    content = []
    if inputs is not None:
        content.append(
            {
                "toolUse": {
                    "toolUseId": "confidence-1",
                    "name": "submit_confidence_evaluation",
                    "input": inputs,
                }
            }
        )
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "tool_use",
        "usage": usage or {"inputTokens": 30, "outputTokens": 10, "totalTokens": 40},
    }


def confidence_inputs(*, unsupported=False, conflict=False, abstention=0.0):
    return {
        "claim_support": 0.9,
        "question_coverage": 1.0,
        "source_consistency": 1.0,
        "evidence_quality": 0.9,
        "abstention_appropriateness": abstention,
        "has_unsupported_material_claim": unsupported,
        "has_unexplained_conflict": conflict,
    }


def hit() -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="handbook.pdf",
        title="Handbook",
        score=0.9,
        text="Two reviewers must approve the procedure.",
        page_numbers=[4],
        heading_path=["Approval"],
    )


def generated(*, status="answered") -> GeneratedAnswer:
    return GeneratedAnswer(
        status=status,
        answer=(
            "Two reviewers are required."
            if status == "answered"
            else "The evidence is insufficient."
        ),
        evidence_ids=("E1",) if status == "answered" else (),
        usage={},
        stop_reason="tool_use",
        attempts=1,
    )


def evaluator(client) -> BedrockRagConfidenceEvaluator:
    return BedrockRagConfidenceEvaluator(
        session=object(),
        model_id="fake-confidence",
        max_output_tokens=500,
        max_evidence_characters=5_000,
        client=client,
    )


def test_answer_score_combines_verifier_and_retrieval_signals() -> None:
    client = FakeBedrockClient([response(confidence_inputs())])

    evaluation = evaluator(client).evaluate(
        "How many reviewers?",
        generated(),
        [hit()],
        evidence_coverage_ratio=1.0,
        retrieval_attempts=1,
    )

    assert evaluation.score == 9.4
    assert evaluation.usage["totalTokens"] == 40
    assert evaluation.warning_reasons() == ()
    request_text = client.requests[0]["messages"][0]["content"][0]["text"]
    assert '"evidence_id": "E1"' in request_text
    assert '"cited": true' in request_text
    assert "Two reviewers must approve" in request_text


def test_unsupported_material_claim_caps_score_at_five() -> None:
    client = FakeBedrockClient(
        [response(confidence_inputs(unsupported=True))]
    )

    evaluation = evaluator(client).evaluate(
        "How many reviewers?",
        generated(),
        [hit()],
        evidence_coverage_ratio=1.0,
        retrieval_attempts=1,
    )

    assert evaluation.score == 5.0
    assert "unsupported_material_claim" in evaluation.warning_reasons()


def test_abstention_score_measures_confidence_in_refusal() -> None:
    client = FakeBedrockClient(
        [response(confidence_inputs(abstention=0.9))]
    )

    evaluation = evaluator(client).evaluate(
        "An unsupported question",
        generated(status="insufficient_evidence"),
        [hit()],
        evidence_coverage_ratio=0.0,
        retrieval_attempts=2,
    )

    assert evaluation.score == 9.3
    assert evaluation.abstention_score == 9.3


def test_malformed_first_response_is_retried_once() -> None:
    client = FakeBedrockClient(
        [response(), response(confidence_inputs())]
    )

    evaluation = evaluator(client).evaluate(
        "How many reviewers?",
        generated(),
        [hit()],
        evidence_coverage_ratio=1.0,
        retrieval_attempts=1,
    )

    assert len(client.requests) == 2
    assert evaluation.usage["totalTokens"] == 80
    retry_text = client.requests[1]["messages"][0]["content"][0]["text"]
    assert "Submit exactly one structured confidence evaluation" in retry_text
