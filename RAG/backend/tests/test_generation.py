from __future__ import annotations

import pytest
from app.generation import AnswerGenerationError, BedrockGroundedAnswerGenerator
from app.models import SearchHit


class FakeBedrockClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def hit() -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="handbook.pdf",
        title="Handbook",
        score=0.91,
        text="Two reviewers must approve the procedure.",
        page_numbers=[4],
        heading_path=["Approval"],
    )


def response(inputs, *, usage=None):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-1",
                            "name": "submit_grounded_answer",
                            "input": inputs,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": usage or {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def generator(client) -> BedrockGroundedAnswerGenerator:
    return BedrockGroundedAnswerGenerator(
        session=object(),
        model_id="fake-generation",
        temperature=0.1,
        max_output_tokens=500,
        max_context_characters=5_000,
        client=client,
    )


def test_grounded_submission_uses_only_valid_evidence() -> None:
    client = FakeBedrockClient(
        [
            response(
                {
                    "status": "answered",
                    "answer": "Two reviewers are required.",
                    "evidence_ids": ["E1"],
                }
            )
        ]
    )

    result = generator(client).generate("How many reviewers?", [hit()])

    assert result.status == "answered"
    assert result.evidence_ids == ("E1",)
    assert result.usage["totalTokens"] == 15
    request = client.requests[0]
    assert request["modelId"] == "fake-generation"
    assert request["toolConfig"]["tools"][0]["toolSpec"]["name"] == "submit_grounded_answer"
    assert "Two reviewers" in request["messages"][0]["content"][0]["text"]


def test_invalid_citation_is_retried_once() -> None:
    client = FakeBedrockClient(
        [
            response(
                {
                    "status": "answered",
                    "answer": "Unsupported.",
                    "evidence_ids": ["E99"],
                }
            ),
            response(
                {
                    "status": "insufficient_evidence",
                    "answer": "The evidence is insufficient.",
                    "evidence_ids": [],
                }
            ),
        ]
    )

    result = generator(client).generate("Unknown?", [hit()])

    assert result.status == "insufficient_evidence"
    assert result.attempts == 2
    assert result.usage["totalTokens"] == 30


def test_model_failure_is_sanitized() -> None:
    class FailureClient:
        def converse(self, **_request):
            raise RuntimeError("secret request body")

    with pytest.raises(AnswerGenerationError, match="RuntimeError") as exc:
        generator(FailureClient()).generate("Question?", [hit()])

    assert "secret request body" not in str(exc.value)
