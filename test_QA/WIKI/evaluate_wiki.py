"""Run the SG-IA ground-truth benchmark against the WIKI HTTP API.

The chatbot and the LLM judge are deliberately separate concerns: questions go
to the running WIKI API, while a separately configured Bedrock model evaluates
point-level correctness and groundedness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import secrets
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import requests


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[1]
DEFAULT_DATASET = HERE.parent / "mateial" / "ground_truth_qa.json"
DEFAULT_WIKI_ROOT = WORKSPACE_ROOT / "WIKI" / "backend" / "wiki"
DEFAULT_CORPUS_MANIFEST = DEFAULT_WIKI_ROOT / ".ingestion-manifest.json"
DEFAULT_AWS_CREDENTIALS = WORKSPACE_ROOT / "WIKI" / "aws_credentials.json"
DEFAULT_PROMPT = HERE / "judge_prompt.md"

POINT_VERDICTS = {"covered", "partially_covered", "missing", "contradicted"}
CLAIM_VERDICTS = {"supported", "unsupported", "contradicted", "not_assessable"}


class EvaluationError(RuntimeError):
    """Raised for configuration, API, judge, or benchmark failures."""


class WikiApiResponseError(EvaluationError):
    """Raised for a non-success response from the WIKI API."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def resolve_config_path(value: str | Path, *, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def validate_benchmark(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
        raise EvaluationError("The benchmark must be an object containing a cases list.")
    if not dataset["cases"]:
        raise EvaluationError("The benchmark contains no cases.")

    seen: set[str] = set()
    for index, case in enumerate(dataset["cases"], start=1):
        if not isinstance(case, dict):
            raise EvaluationError(f"Benchmark case {index} is not an object.")
        case_id = str(case.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise EvaluationError(f"Benchmark case {index} has a missing or duplicate id.")
        seen.add(case_id)
        for field in ("question", "expected_status", "ground_truth_answer"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise EvaluationError(f"Case {case_id} has an invalid {field}.")
        if case["expected_status"] not in {"answered", "insufficient_knowledge"}:
            raise EvaluationError(f"Case {case_id} has an unsupported expected_status.")
        points = case.get("required_answer_points")
        if not isinstance(points, list) or not points or not all(isinstance(p, str) for p in points):
            raise EvaluationError(f"Case {case_id} has invalid required_answer_points.")
        if not isinstance(case.get("sources", []), list):
            raise EvaluationError(f"Case {case_id} has invalid sources.")
    return dataset


class WikiApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 240.0,
        chat_max_attempts: int = 3,
        chat_retry_delay_seconds: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = (5.0, timeout_seconds)
        self.chat_max_attempts = chat_max_attempts
        self.chat_retry_delay_seconds = chat_retry_delay_seconds
        self.last_chat_attempts = 0
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise EvaluationError(f"Cannot reach the WIKI API at {self.base_url}: {exc}") from exc
        if not response.ok:
            try:
                payload = response.json()
                detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            except ValueError:
                detail = response.text.strip() or "No response body"
            raise WikiApiResponseError(
                f"WIKI API {path} failed ({response.status_code}): {detail}",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise EvaluationError(f"WIKI API {path} returned invalid JSON.") from exc

    def health(self) -> dict[str, Any]:
        value = self._request("GET", "/health", timeout=(3.0, 30.0))
        if not isinstance(value, dict):
            raise EvaluationError("WIKI /health returned an invalid response.")
        return value

    def lint(self) -> dict[str, Any]:
        value = self._request("GET", "/wiki/lint", timeout=(3.0, 60.0))
        if not isinstance(value, dict):
            raise EvaluationError("WIKI /wiki/lint returned an invalid response.")
        return value

    def chat(self, question: str) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        value: Any = None
        for attempt in range(1, self.chat_max_attempts + 1):
            self.last_chat_attempts = attempt
            try:
                value = self._request("POST", "/chat", json={"question": question})
                break
            except WikiApiResponseError as exc:
                retryable = exc.status_code in {429, 500, 502, 503, 504}
                if not retryable or attempt == self.chat_max_attempts:
                    raise
                time.sleep(self.chat_retry_delay_seconds * (2 ** (attempt - 1)))
        client_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            raise EvaluationError("WIKI /chat returned an invalid response.")
        return value, client_latency_ms


JUDGE_TOOL = {
    "toolSpec": {
        "name": "submit_evaluation",
        "description": "Submit the complete structured evaluation of one chatbot answer.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "point_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "point_index": {"type": "integer", "minimum": 1},
                                "verdict": {
                                    "type": "string",
                                    "enum": sorted(POINT_VERDICTS),
                                },
                                "explanation": {"type": "string"},
                            },
                            "required": ["point_index", "verdict", "explanation"],
                        },
                    },
                    "correctness_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "correctness_explanation": {"type": "string"},
                    "missing_information": {"type": "array", "items": {"type": "string"}},
                    "incorrect_claims": {"type": "array", "items": {"type": "string"}},
                    "claim_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "verdict": {
                                    "type": "string",
                                    "enum": sorted(CLAIM_VERDICTS),
                                },
                                "evidence_paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "explanation": {"type": "string"},
                            },
                            "required": ["claim", "verdict", "evidence_paths", "explanation"],
                        },
                    },
                    "groundedness_evaluated": {"type": "boolean"},
                    "groundedness_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "point_results",
                    "correctness_score",
                    "correctness_explanation",
                    "missing_information",
                    "incorrect_claims",
                    "claim_results",
                    "groundedness_evaluated",
                    "groundedness_score",
                    "unsupported_claims",
                ],
            }
        },
    }
}


@dataclass(frozen=True)
class JudgeResponse:
    evaluation: dict[str, Any]
    usage: dict[str, int]
    latency_ms: float
    stop_reason: str
    structured_mode: str


class BedrockJudge:
    def __init__(
        self,
        *,
        model_id: str,
        region_name: str | None,
        system_prompt: str,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
        credentials_file: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.credentials_file = credentials_file
        self._client = client

    def _make_client(self) -> Any:
        try:
            import boto3  # type: ignore
            from botocore.config import Config  # type: ignore
        except ImportError as exc:
            raise EvaluationError(
                "Boto3 is required for the Bedrock judge. Install requirements.txt."
            ) from exc

        session_kwargs: dict[str, str] = {}
        client_region = self.region_name
        if self.credentials_file and self.credentials_file.is_file():
            values = load_json(self.credentials_file)
            if not isinstance(values, dict):
                raise EvaluationError("The AWS credentials file must contain a JSON object.")
            access_key = str(values.get("aws_access_key_id", "")).strip()
            secret_key = str(values.get("aws_secret_access_key", "")).strip()
            session_token = str(values.get("aws_session_token", "")).strip()
            if access_key:
                if not secret_key:
                    raise EvaluationError("AWS access key is present but its secret key is missing.")
                session_kwargs.update(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                if session_token:
                    session_kwargs["aws_session_token"] = session_token
            if not client_region:
                client_region = str(values.get("region_name", "")).strip() or None

        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {
            "config": Config(
                connect_timeout=10,
                read_timeout=300,
                retries={"max_attempts": 4, "mode": "standard"},
            )
        }
        if client_region:
            client_kwargs["region_name"] = client_region
        self.region_name = client_region
        return session.client("bedrock-runtime", **client_kwargs)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def evaluate(self, payload: Mapping[str, Any]) -> JudgeResponse:
        user_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        inference_config: dict[str, Any] = {"maxTokens": self.max_output_tokens}
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature
        base_request: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": self.system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": inference_config,
        }

        request_variants = [
            (
                "forced_tool",
                {
                    **base_request,
                    "toolConfig": {
                        "tools": [JUDGE_TOOL],
                        "toolChoice": {"tool": {"name": "submit_evaluation"}},
                    },
                },
            ),
            ("tool", {**base_request, "toolConfig": {"tools": [JUDGE_TOOL]}}),
            ("json_text", base_request),
        ]

        started = time.perf_counter()
        last_validation_error: Exception | None = None
        response: Mapping[str, Any] | None = None
        mode = "unknown"
        for mode, request in request_variants:
            try:
                candidate = self.client.converse(**request)
                if not isinstance(candidate, Mapping):
                    raise EvaluationError("Bedrock returned a non-object response.")
                response = candidate
                break
            except Exception as exc:
                error_response = getattr(exc, "response", None)
                error = error_response.get("Error", {}) if isinstance(error_response, Mapping) else {}
                code = error.get("Code") if isinstance(error, Mapping) else None
                if code == "ValidationException":
                    last_validation_error = exc
                    continue
                suffix = f" ({code})" if code else ""
                raise EvaluationError(
                    f"Bedrock judge request failed: {type(exc).__name__}{suffix}."
                ) from exc
        if response is None:
            raise EvaluationError(
                "The judge model rejected forced tools, tools, and JSON-text evaluation."
            ) from last_validation_error

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            content = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise EvaluationError("Bedrock judge returned an invalid response shape.") from exc
        if not isinstance(content, list):
            raise EvaluationError("Bedrock judge returned invalid message content.")

        evaluation: dict[str, Any] | None = None
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            tool_use = block.get("toolUse")
            if isinstance(tool_use, Mapping) and tool_use.get("name") == "submit_evaluation":
                value = tool_use.get("input")
                if isinstance(value, Mapping):
                    evaluation = dict(value)
                    break
            if isinstance(block.get("text"), str):
                text_parts.append(str(block["text"]))
        if evaluation is None and text_parts:
            evaluation = parse_json_object("\n".join(text_parts))
        if evaluation is None:
            raise EvaluationError("Bedrock judge did not return a structured evaluation.")

        validated = validate_judgment(evaluation, payload.get("required_answer_points", []))
        usage_raw = response.get("usage", {})
        usage = {
            str(key): int(value)
            for key, value in usage_raw.items()
            if isinstance(value, int)
        } if isinstance(usage_raw, Mapping) else {}
        return JudgeResponse(
            evaluation=validated,
            usage=usage,
            latency_ms=latency_ms,
            stop_reason=str(response.get("stopReason", "unknown")),
            structured_mode=mode,
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"Judge field {field} must be a list of strings.")
    return [item.strip() for item in value if item.strip()]


def validate_judgment(value: Mapping[str, Any], required_points: Any) -> dict[str, Any]:
    if not isinstance(required_points, list):
        raise EvaluationError("Judge payload has invalid required_answer_points.")
    point_results = value.get("point_results")
    if not isinstance(point_results, list):
        raise EvaluationError("Judge result is missing point_results.")
    normalized_points: dict[int, dict[str, Any]] = {}
    for item in point_results:
        if not isinstance(item, Mapping):
            raise EvaluationError("Judge returned an invalid point result.")
        index = item.get("point_index")
        verdict = item.get("verdict")
        if not isinstance(index, int) or index < 1 or index > len(required_points):
            raise EvaluationError("Judge returned an invalid point_index.")
        if index in normalized_points or verdict not in POINT_VERDICTS:
            raise EvaluationError("Judge returned duplicate points or an invalid point verdict.")
        normalized_points[index] = {
            "point_index": index,
            "point": required_points[index - 1],
            "verdict": verdict,
            "explanation": str(item.get("explanation", "")).strip(),
        }
    if set(normalized_points) != set(range(1, len(required_points) + 1)):
        raise EvaluationError("Judge did not evaluate every required answer point exactly once.")

    score = value.get("correctness_score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise EvaluationError("Judge correctness_score must be an integer from 1 to 5.")
    grounded = value.get("groundedness_evaluated")
    grounding_score = value.get("groundedness_score")
    if not isinstance(grounded, bool) or not isinstance(grounding_score, (int, float)):
        raise EvaluationError("Judge returned invalid groundedness fields.")
    grounding_score = float(grounding_score)
    if not 0 <= grounding_score <= 1:
        raise EvaluationError("Judge groundedness_score must be between 0 and 1.")

    claims_raw = value.get("claim_results")
    if not isinstance(claims_raw, list):
        raise EvaluationError("Judge claim_results must be a list.")
    claims: list[dict[str, Any]] = []
    for item in claims_raw:
        if not isinstance(item, Mapping) or item.get("verdict") not in CLAIM_VERDICTS:
            raise EvaluationError("Judge returned an invalid claim result.")
        claims.append(
            {
                "claim": str(item.get("claim", "")).strip(),
                "verdict": item["verdict"],
                "evidence_paths": string_list(item.get("evidence_paths", []), "evidence_paths"),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )

    return {
        "point_results": [normalized_points[index] for index in sorted(normalized_points)],
        "correctness_score": score,
        "correctness_explanation": str(value.get("correctness_explanation", "")).strip(),
        "missing_information": string_list(value.get("missing_information"), "missing_information"),
        "incorrect_claims": string_list(value.get("incorrect_claims"), "incorrect_claims"),
        "claim_results": claims,
        "groundedness_evaluated": grounded,
        "groundedness_score": grounding_score if grounded else None,
        "unsupported_claims": string_list(value.get("unsupported_claims"), "unsupported_claims"),
    }


def normalize_source_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    for prefix in ("WIKI/backend/", "backend/"):
        if path.casefold().startswith(prefix.casefold()):
            path = path[len(prefix) :]
    return path.casefold()


def citation_metrics(case: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        normalize_source_path(item.get("source_path"))
        for item in case.get("sources", [])
        if isinstance(item, Mapping) and item.get("source_path")
    }
    actual: set[str] = set()
    citations = response.get("citations", [])
    if isinstance(citations, list):
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            paths = citation.get("source_paths", [])
            if isinstance(paths, list):
                actual.update(normalize_source_path(path) for path in paths if path)
    matched = expected & actual
    return {
        "expected_sources": sorted(expected),
        "cited_sources": sorted(actual),
        "matched_expected_sources": sorted(matched),
        "expected_source_recall": round(len(matched) / len(expected), 6) if expected else None,
        "expected_source_precision": round(len(matched) / len(actual), 6) if actual else None,
    }


def wiki_paths_from_response(response: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    debug = response.get("debug", {})
    if isinstance(debug, Mapping) and isinstance(debug.get("pages_read"), list):
        paths.extend(str(item) for item in debug["pages_read"] if item)
    citations = response.get("citations", [])
    if isinstance(citations, list):
        for item in citations:
            if isinstance(item, Mapping) and item.get("wiki_path"):
                paths.append(str(item["wiki_path"]))
    return list(dict.fromkeys(paths))


def safe_wiki_file(wiki_root: Path, wiki_path: str) -> Path | None:
    normalized = wiki_path.strip().replace("\\", "/")
    if normalized.casefold().startswith("wiki/"):
        normalized = normalized[5:]
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts or not normalized:
        return None
    root = wiki_root.resolve()
    candidate = root.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def collect_wiki_evidence(
    wiki_root: Path,
    wiki_paths: Iterable[str],
    *,
    max_chars_per_page: int,
    max_total_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judge_pages: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    remaining = max_total_chars
    for wiki_path in wiki_paths:
        if remaining <= 0:
            break
        path = safe_wiki_file(wiki_root, wiki_path)
        if path is None:
            metadata.append({"wiki_path": wiki_path, "available": False})
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        limit = min(max_chars_per_page, remaining)
        included = content[:limit]
        truncated = len(included) < len(content)
        relative = path.relative_to(wiki_root.resolve()).as_posix()
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        judge_pages.append(
            {
                "wiki_path": relative,
                "content": included,
                "truncated": truncated,
            }
        )
        metadata.append(
            {
                "wiki_path": relative,
                "available": True,
                "sha256": digest,
                "original_characters": len(content),
                "included_characters": len(included),
                "truncated": truncated,
            }
        )
        remaining -= len(included)
    return judge_pages, metadata


def point_coverage(point_results: Sequence[Mapping[str, Any]]) -> float:
    if not point_results:
        return 0.0
    weights = {"covered": 1.0, "partially_covered": 0.5, "missing": 0.0, "contradicted": 0.0}
    return round(sum(weights.get(str(item.get("verdict")), 0.0) for item in point_results) / len(point_results), 6)


def classify_result(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    judgment: Mapping[str, Any],
    citations: Mapping[str, Any],
) -> tuple[str, list[str]]:
    expected_status = str(case.get("expected_status"))
    actual_status = str(response.get("status", "answered"))
    score = int(judgment["correctness_score"])
    unsupported = bool(judgment.get("unsupported_claims")) or bool(judgment.get("incorrect_claims")) or any(
        item.get("verdict") in {"unsupported", "contradicted"}
        for item in judgment.get("claim_results", [])
    )
    flags: list[str] = []
    if unsupported:
        flags.append("HALLUCINATION")

    if expected_status == "insufficient_knowledge":
        primary = (
            "EXPECTED_ABSTENTION"
            if actual_status == "insufficient_knowledge" and score >= 4 and not unsupported
            else "INCORRECT"
        )
        if actual_status != "insufficient_knowledge":
            flags.append("ABSTENTION_FAILURE")
        return primary, flags

    if actual_status == "insufficient_knowledge":
        return "FALSE_ABSTENTION", flags
    primary = "CORRECT" if score >= 4 else "PARTIALLY_CORRECT" if score == 3 else "INCORRECT"

    recall = citations.get("expected_source_recall")
    if score <= 3:
        if recall is None or float(recall) == 0:
            flags.append("WIKI_LOOKUP_FAILURE")
        else:
            flags.append("ANSWER_GENERATION_FAILURE")
    if not response.get("citations"):
        flags.append("MISSING_CITATION")
    elif recall is not None and float(recall) < 1:
        flags.append("WRONG_OR_INCOMPLETE_CITATION")
    return primary, list(dict.fromkeys(flags))


def usage_tokens(usage: Mapping[str, Any], direction: str) -> int:
    candidates = (
        ("inputTokens", "input_tokens", "prompt_tokens")
        if direction == "input"
        else ("outputTokens", "output_tokens", "completion_tokens")
    )
    for key in candidates:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def estimate_cost(usage: Mapping[str, Any], input_price: float | None, output_price: float | None) -> float | None:
    if input_price is None or output_price is None:
        return None
    value = (
        usage_tokens(usage, "input") * input_price
        + usage_tokens(usage, "output") * output_price
    ) / 1_000_000
    return round(value, 8)


def result_costs(record: Mapping[str, Any], pricing: Mapping[str, Any]) -> dict[str, float | None]:
    chatbot = record.get("chatbot", {})
    chatbot_usage = chatbot.get("usage", {}) if isinstance(chatbot, Mapping) else {}
    judge = record.get("judge", {})
    judge_usage = judge.get("usage", {}) if isinstance(judge, Mapping) else {}
    chatbot_cost = estimate_cost(
        chatbot_usage if isinstance(chatbot_usage, Mapping) else {},
        pricing.get("chatbot_input_per_million_usd"),
        pricing.get("chatbot_output_per_million_usd"),
    )
    judge_cost = estimate_cost(
        judge_usage if isinstance(judge_usage, Mapping) else {},
        pricing.get("judge_input_per_million_usd"),
        pricing.get("judge_output_per_million_usd"),
    )
    total = (
        round(chatbot_cost + judge_cost, 8)
        if chatbot_cost is not None and judge_cost is not None
        else None
    )
    return {"chatbot": chatbot_cost, "judge": judge_cost, "total": total}


def percentile(values: Sequence[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def average(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.fmean(selected), 6) if selected else None


def summarize(results: Sequence[Mapping[str, Any]], pricing: Mapping[str, Any]) -> dict[str, Any]:
    judged = [item for item in results if isinstance(item.get("judgment"), Mapping)]
    api_successes = [item for item in results if isinstance(item.get("chatbot"), Mapping)]
    answerable = [item for item in judged if item.get("expected_status") == "answered"]
    negatives = [item for item in judged if item.get("expected_status") == "insufficient_knowledge"]
    scores = [float(item["judgment"]["correctness_score"]) for item in judged]
    grounded = [
        float(item["judgment"]["groundedness_score"])
        for item in judged
        if item["judgment"].get("groundedness_evaluated")
        and isinstance(item["judgment"].get("groundedness_score"), (int, float))
    ]
    server_latencies = [
        float(item["chatbot"].get("latency_ms", 0)) for item in api_successes
        if isinstance(item["chatbot"].get("latency_ms"), (int, float))
    ]
    client_latencies = [float(item["client_latency_ms"]) for item in api_successes]

    chatbot_usage: dict[str, int] = {}
    judge_usage: dict[str, int] = {}
    for item in api_successes:
        for key, value in item["chatbot"].get("usage", {}).items():
            if isinstance(value, int):
                chatbot_usage[str(key)] = chatbot_usage.get(str(key), 0) + value
    for item in judged:
        for key, value in item.get("judge", {}).get("usage", {}).items():
            if isinstance(value, int):
                judge_usage[str(key)] = judge_usage.get(str(key), 0) + value

    outcomes: dict[str, int] = {}
    for item in results:
        outcome = str(item.get("primary_outcome", "UNKNOWN"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    hallucinations = sum("HALLUCINATION" in item.get("diagnostic_flags", []) for item in judged)
    score_denominator = len(judged) or 1
    answerable_denominator = len(answerable) or 1
    negative_denominator = len(negatives) or 1

    chatbot_cost = estimate_cost(
        chatbot_usage,
        pricing.get("chatbot_input_per_million_usd"),
        pricing.get("chatbot_output_per_million_usd"),
    )
    judge_cost = estimate_cost(
        judge_usage,
        pricing.get("judge_input_per_million_usd"),
        pricing.get("judge_output_per_million_usd"),
    )
    total_cost = round(chatbot_cost + judge_cost, 8) if chatbot_cost is not None and judge_cost is not None else None

    def group_breakdown(field: str) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for item in judged:
            groups.setdefault(str(item.get(field) or "unknown"), []).append(item)
        output: dict[str, Any] = {}
        for name, items in sorted(groups.items()):
            group_scores = [float(item["judgment"]["correctness_score"]) for item in items]
            output[name] = {
                "evaluations": len(items),
                "average_correctness_score": average(group_scores),
                "percentage_score_at_least_4": round(
                    100 * sum(score >= 4 for score in group_scores) / len(group_scores), 2
                ),
                "average_required_point_coverage": average(
                    item.get("required_point_coverage") for item in items
                ),
                "average_groundedness": average(
                    item["judgment"].get("groundedness_score")
                    for item in items
                    if item["judgment"].get("groundedness_evaluated")
                ),
                "average_expected_source_recall": average(
                    item.get("citation_metrics", {}).get("expected_source_recall")
                    for item in items
                ),
            }
        return output

    return {
        "total_evaluations": len(results),
        "unique_questions": len({item.get("case_id") for item in results}),
        "api_successes": len(api_successes),
        "api_errors": sum(item.get("primary_outcome") == "API_ERROR" for item in results),
        "judge_successes": len(judged),
        "judge_errors": sum(item.get("primary_outcome") == "JUDGE_ERROR" for item in results),
        "outcome_counts": outcomes,
        "correctness": {
            "average_score_1_to_5": average(scores),
            "average_required_point_coverage": average(item.get("required_point_coverage") for item in answerable),
            "percentage_score_at_least_4": round(100 * sum(score >= 4 for score in scores) / score_denominator, 2),
            "percentage_fully_correct": round(100 * sum(score == 5 for score in scores) / score_denominator, 2),
            "percentage_partially_correct": round(100 * sum(score == 3 for score in scores) / score_denominator, 2),
            "percentage_incorrect": round(100 * sum(score <= 2 for score in scores) / score_denominator, 2),
        },
        "grounding_and_sources": {
            "average_groundedness": average(grounded),
            "groundedness_evaluated_count": len(grounded),
            "average_expected_source_recall": average(
                item.get("citation_metrics", {}).get("expected_source_recall") for item in answerable
            ),
            "hallucination_rate_percent": round(100 * hallucinations / score_denominator, 2),
        },
        "abstention": {
            "unanswerable_evaluations": len(negatives),
            "successful_abstentions": sum(item.get("primary_outcome") == "EXPECTED_ABSTENTION" for item in negatives),
            "successful_abstention_rate_percent": round(
                100 * sum(item.get("primary_outcome") == "EXPECTED_ABSTENTION" for item in negatives) / negative_denominator,
                2,
            ),
            "false_abstentions_on_answerable": sum(item.get("primary_outcome") == "FALSE_ABSTENTION" for item in answerable),
            "false_abstention_rate_percent": round(
                100 * sum(item.get("primary_outcome") == "FALSE_ABSTENTION" for item in answerable) / answerable_denominator,
                2,
            ),
        },
        "latency_ms": {
            "server_average": average(server_latencies),
            "server_median": percentile(server_latencies, 0.5),
            "server_p95": percentile(server_latencies, 0.95),
            "client_average": average(client_latencies),
            "client_median": percentile(client_latencies, 0.5),
            "client_p95": percentile(client_latencies, 0.95),
        },
        "usage": {"chatbot": chatbot_usage, "judge": judge_usage},
        "estimated_cost_usd": {
            "chatbot": chatbot_cost,
            "judge": judge_cost,
            "total": total_cost,
            "note": "null means pricing was not configured; Bedrock billing is not inferred.",
        },
        "by_question_type": group_breakdown("question_type"),
        "by_difficulty": group_breakdown("difficulty"),
    }


def judge_payload(
    case: Mapping[str, Any],
    response: Mapping[str, Any],
    evidence_pages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "evaluation_language": "it",
        "question": case["question"],
        "expected_status": case["expected_status"],
        "ground_truth_answer": case["ground_truth_answer"],
        "required_answer_points": list(case["required_answer_points"]),
        "reference_sources": list(case.get("sources", [])),
        "chatbot_status": response.get("status"),
        "chatbot_answer": response.get("answer"),
        "chatbot_citations": response.get("citations", []),
        "chatbot_debug": response.get("debug", {}),
        "consulted_wiki_pages": list(evidence_pages),
    }


def case_csv_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        judgment = item.get("judgment", {}) if isinstance(item.get("judgment"), Mapping) else {}
        chatbot = item.get("chatbot", {}) if isinstance(item.get("chatbot"), Mapping) else {}
        citations = item.get("citation_metrics", {}) if isinstance(item.get("citation_metrics"), Mapping) else {}
        rows.append(
            {
                "case_id": item.get("case_id"),
                "repetition": item.get("repetition"),
                "question_type": item.get("question_type"),
                "difficulty": item.get("difficulty"),
                "expected_status": item.get("expected_status"),
                "chatbot_status": chatbot.get("status"),
                "correctness_score": judgment.get("correctness_score"),
                "required_point_coverage": item.get("required_point_coverage"),
                "groundedness_score": judgment.get("groundedness_score"),
                "expected_source_recall": citations.get("expected_source_recall"),
                "primary_outcome": item.get("primary_outcome"),
                "diagnostic_flags": "|".join(item.get("diagnostic_flags", [])),
                "server_latency_ms": chatbot.get("latency_ms"),
                "client_latency_ms": item.get("client_latency_ms"),
                "chatbot_input_tokens": usage_tokens(chatbot.get("usage", {}), "input"),
                "chatbot_output_tokens": usage_tokens(chatbot.get("usage", {}), "output"),
                "judge_input_tokens": usage_tokens(item.get("judge", {}).get("usage", {}), "input"),
                "judge_output_tokens": usage_tokens(item.get("judge", {}).get("usage", {}), "output"),
                "chatbot_estimated_cost_usd": item.get("estimated_cost_usd", {}).get("chatbot"),
                "judge_estimated_cost_usd": item.get("estimated_cost_usd", {}).get("judge"),
                "total_estimated_cost_usd": item.get("estimated_cost_usd", {}).get("total"),
                "error": item.get("error"),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file (default: config.json if present).")
    parser.add_argument("--api-url", help="WIKI API base URL.")
    parser.add_argument("--dataset", type=Path, help="Ground-truth benchmark JSON.")
    parser.add_argument("--wiki-root", type=Path, help="Directory containing generated Wiki Markdown.")
    parser.add_argument("--output-dir", type=Path, help="Parent directory for timestamped run results.")
    parser.add_argument("--judge-model-id", help="Bedrock model or inference-profile ID for the judge.")
    parser.add_argument("--aws-region", help="AWS region for the Bedrock judge.")
    parser.add_argument("--aws-credentials-file", type=Path, help="Optional local AWS credential JSON.")
    parser.add_argument("--repetitions", type=int, help="Independent chatbot answers per question.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only this case ID; repeat as needed.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip /health and /wiki/lint validation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_config = HERE / "config.json"
    config_path = args.config.resolve() if args.config else (default_config if default_config.is_file() else None)
    config: dict[str, Any] = {}
    config_dir = HERE
    if config_path:
        loaded = load_json(config_path)
        if not isinstance(loaded, dict):
            raise EvaluationError("The evaluator config must contain a JSON object.")
        config = loaded
        config_dir = config_path.parent

    api_url = args.api_url or os.getenv("LLM_WIKI_API_URL") or config.get("api_url") or "http://127.0.0.1:8000"
    dataset_path = args.dataset or resolve_config_path(config.get("dataset", str(DEFAULT_DATASET)), config_dir=config_dir)
    wiki_root = args.wiki_root or resolve_config_path(config.get("wiki_root", str(DEFAULT_WIKI_ROOT)), config_dir=config_dir)
    output_parent = args.output_dir or resolve_config_path(config.get("output_dir", "results"), config_dir=config_dir)
    credentials_value = args.aws_credentials_file or nested(config, "judge", "aws_credentials_file")
    credentials_file = (
        resolve_config_path(credentials_value, config_dir=config_dir)
        if credentials_value
        else DEFAULT_AWS_CREDENTIALS if DEFAULT_AWS_CREDENTIALS.is_file() else None
    )
    judge_model_id = (
        args.judge_model_id
        or os.getenv("BEDROCK_JUDGE_MODEL_ID")
        or nested(config, "judge", "model_id")
    )
    if not judge_model_id:
        raise EvaluationError(
            "A separate judge model is required. Set BEDROCK_JUDGE_MODEL_ID, "
            "--judge-model-id, or judge.model_id in config.json."
        )
    aws_region = (
        args.aws_region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or nested(config, "judge", "region_name")
    )
    repetitions = args.repetitions or int(nested(config, "run", "repetitions", default=1))
    if repetitions < 1:
        raise EvaluationError("repetitions must be at least 1.")
    max_output_tokens = int(nested(config, "judge", "max_output_tokens", default=4096))
    raw_temperature = nested(config, "judge", "temperature", default=None)
    temperature = float(raw_temperature) if raw_temperature is not None else None
    if temperature not in {None, 0.0}:
        print("Warning: judge temperature is non-zero; deterministic judging is recommended.", file=sys.stderr)

    dataset = validate_benchmark(load_json(Path(dataset_path)))
    if not Path(wiki_root).is_dir():
        raise EvaluationError(f"Wiki root does not exist or is not a directory: {wiki_root}")
    cases = list(dataset["cases"])
    if args.case_id:
        selected = set(args.case_id)
        unknown = selected - {str(case["id"]) for case in cases}
        if unknown:
            raise EvaluationError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in selected]
    if args.limit is not None:
        if args.limit < 1:
            raise EvaluationError("--limit must be at least 1.")
        cases = cases[: args.limit]

    prompt_path = resolve_config_path(nested(config, "judge", "prompt", default=str(DEFAULT_PROMPT)), config_dir=config_dir)
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise EvaluationError(f"Judge prompt not found: {prompt_path}") from exc

    wiki_client = WikiApiClient(
        str(api_url),
        timeout_seconds=float(nested(config, "run", "chat_timeout_seconds", default=240)),
        chat_max_attempts=int(nested(config, "run", "chat_max_attempts", default=3)),
        chat_retry_delay_seconds=float(
            nested(config, "run", "chat_retry_delay_seconds", default=1.5)
        ),
    )
    health: dict[str, Any] | None = None
    lint: dict[str, Any] | None = None
    if not args.skip_preflight:
        health = wiki_client.health()
        if not health.get("bedrock_configured"):
            raise EvaluationError("The WIKI backend is online but Bedrock is not configured.")
        lint = wiki_client.lint()
        if lint.get("valid") is not True:
            raise EvaluationError("The WIKI lint check failed; fix the knowledge base before benchmarking.")
        if str(health.get("model_id", "")) == str(judge_model_id):
            print(
                "Warning: the chatbot and judge use the same model ID; an independent judge is preferred.",
                file=sys.stderr,
            )

    judge = BedrockJudge(
        model_id=str(judge_model_id),
        region_name=str(aws_region) if aws_region else None,
        system_prompt=system_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        credentials_file=credentials_file,
    )
    max_chars_per_page = int(nested(config, "run", "max_evidence_chars_per_page", default=12000))
    max_total_chars = int(nested(config, "run", "max_total_evidence_chars", default=60000))
    judge_max_attempts = int(nested(config, "run", "judge_max_attempts", default=2))
    if judge_max_attempts < 1:
        raise EvaluationError("judge_max_attempts must be at least 1.")
    pricing = nested(config, "pricing", default={})
    if not isinstance(pricing, Mapping):
        raise EvaluationError("pricing must be a JSON object.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    run_dir = Path(output_parent).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.jsonl"
    started_at = utc_now()
    results: list[dict[str, Any]] = []

    total = len(cases) * repetitions
    counter = 0
    print(f"Run {run_id}: {len(cases)} question(s) x {repetitions} repetition(s)")
    for case in cases:
        for repetition in range(1, repetitions + 1):
            counter += 1
            print(f"[{counter}/{total}] {case['id']} repetition {repetition}", flush=True)
            record: dict[str, Any] = {
                "case_id": case["id"],
                "repetition": repetition,
                "question": case["question"],
                "question_type": case.get("question_type"),
                "difficulty": case.get("difficulty"),
                "expected_status": case["expected_status"],
                "ground_truth_answer": case["ground_truth_answer"],
                "required_answer_points": case["required_answer_points"],
                "reference_sources": case.get("sources", []),
                "started_at": utc_now(),
            }
            try:
                response, client_latency = wiki_client.chat(str(case["question"]))
                record["chatbot"] = response
                record["client_latency_ms"] = client_latency
                record["chatbot_attempts"] = wiki_client.last_chat_attempts
            except EvaluationError as exc:
                record.update(
                    error=str(exc),
                    primary_outcome="API_ERROR",
                    diagnostic_flags=["API_ERROR"],
                    finished_at=utc_now(),
                )
                record["estimated_cost_usd"] = result_costs(record, pricing)
                results.append(record)
                with results_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            metrics = citation_metrics(case, response)
            record["citation_metrics"] = metrics
            evidence_pages, evidence_metadata = collect_wiki_evidence(
                Path(wiki_root),
                wiki_paths_from_response(response),
                max_chars_per_page=max_chars_per_page,
                max_total_chars=max_total_chars,
            )
            record["wiki_evidence"] = evidence_metadata
            try:
                judged: JudgeResponse | None = None
                last_judge_error: EvaluationError | None = None
                for judge_attempt in range(1, judge_max_attempts + 1):
                    record["judge_attempts"] = judge_attempt
                    try:
                        judged = judge.evaluate(judge_payload(case, response, evidence_pages))
                        break
                    except EvaluationError as exc:
                        last_judge_error = exc
                if judged is None:
                    raise last_judge_error or EvaluationError("Judge evaluation failed.")
                record["judgment"] = judged.evaluation
                record["judge"] = {
                    "model_id": judge.model_id,
                    "usage": judged.usage,
                    "latency_ms": judged.latency_ms,
                    "stop_reason": judged.stop_reason,
                    "structured_mode": judged.structured_mode,
                }
                record["required_point_coverage"] = point_coverage(judged.evaluation["point_results"])
                primary, flags = classify_result(case, response, judged.evaluation, metrics)
                record["primary_outcome"] = primary
                record["diagnostic_flags"] = flags
            except EvaluationError as exc:
                record.update(
                    error=str(exc),
                    primary_outcome="JUDGE_ERROR",
                    diagnostic_flags=["JUDGE_ERROR"],
                )
            record["estimated_cost_usd"] = result_costs(record, pricing)
            record["finished_at"] = utc_now()
            results.append(record)
            with results_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(results, pricing)
    finished_at = utc_now()
    write_json(run_dir / "summary.json", summary)
    rows = case_csv_rows(results)
    with (run_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    corpus_manifest_value = dataset.get("corpus_manifest")
    corpus_manifest = (
        (WORKSPACE_ROOT / str(corpus_manifest_value)).resolve()
        if corpus_manifest_value
        else DEFAULT_CORPUS_MANIFEST
    )
    manifest = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "api_url": api_url,
        "health": health,
        "wiki_lint": {
            "valid": lint.get("valid"),
            "pages_checked": lint.get("pages_checked"),
            "graph": lint.get("graph"),
        } if lint else None,
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": sha256_file(Path(dataset_path)),
            "schema_version": dataset.get("schema_version"),
            "selected_case_ids": [case["id"] for case in cases],
        },
        "corpus_manifest": {
            "path": str(corpus_manifest),
            "sha256": sha256_file(corpus_manifest),
        },
        "wiki_root": str(Path(wiki_root).resolve()),
        "judge": {
            "model_id": judge.model_id,
            "region_name": judge.region_name,
            "temperature": judge.temperature,
            "max_output_tokens": judge.max_output_tokens,
            "prompt_path": str(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
        },
        "run": {
            "repetitions": repetitions,
            "max_evidence_chars_per_page": max_chars_per_page,
            "max_total_evidence_chars": max_total_chars,
            "chat_max_attempts": wiki_client.chat_max_attempts,
            "chat_retry_delay_seconds": wiki_client.chat_retry_delay_seconds,
            "judge_max_attempts": judge_max_attempts,
        },
        "pricing": dict(pricing),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"Completed. Results: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["api_errors"] and not summary["judge_errors"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(2)
