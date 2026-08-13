from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import SearchHit


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "ad",
        "al",
        "alla",
        "alle",
        "anche",
        "che",
        "chi",
        "come",
        "con",
        "cosa",
        "da",
        "dal",
        "dalla",
        "dalle",
        "dei",
        "del",
        "della",
        "delle",
        "di",
        "dopo",
        "dove",
        "do",
        "does",
        "e",
        "gli",
        "ha",
        "i",
        "il",
        "in",
        "indica",
        "indicano",
        "indicati",
        "is",
        "la",
        "le",
        "lo",
        "many",
        "mean",
        "means",
        "nei",
        "nel",
        "nella",
        "nelle",
        "o",
        "of",
        "per",
        "for",
        "from",
        "puo",
        "quale",
        "quali",
        "quando",
        "sono",
        "su",
        "the",
        "to",
        "tra",
        "un",
        "una",
        "vale",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "are",
        "was",
        "were",
        "list",
    }
)


def normalized_tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(character for character in folded if not unicodedata.combining(character))
    return _TOKEN_RE.findall(ascii_value)


def question_facets(question: str, *, maximum: int = 20) -> tuple[str, ...]:
    facets: list[str] = []
    for token in normalized_tokens(question):
        if token in _STOPWORDS or (len(token) < 3 and not token.isdigit()):
            continue
        if token not in facets:
            facets.append(token)
    return tuple(facets[:maximum])


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    facets: tuple[str, ...]
    covered_facets: tuple[str, ...]
    missing_facets: tuple[str, ...]
    ratio: float
    sufficient: bool


def assess_coverage(
    question: str,
    evidence: Sequence[SearchHit],
    *,
    minimum_ratio: float,
) -> CoverageAssessment:
    facets = question_facets(question)
    evidence_tokens = {
        token
        for hit in evidence
        for token in normalized_tokens(
            " ".join([hit.filename, hit.title, " ".join(hit.heading_path), hit.text])
        )
    }
    covered = tuple(facet for facet in facets if facet in evidence_tokens)
    missing = tuple(facet for facet in facets if facet not in evidence_tokens)
    ratio = len(covered) / len(facets) if facets else 1.0
    numeric_missing = any(facet.isdigit() for facet in missing)
    return CoverageAssessment(
        facets=facets,
        covered_facets=covered,
        missing_facets=missing,
        ratio=round(ratio, 6),
        sufficient=ratio >= minimum_ratio and not numeric_missing,
    )


def merge_hits(*groups: Iterable[SearchHit]) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    for group in groups:
        for hit in group:
            current = merged.get(hit.chunk_id)
            if current is None or hit.score > current.score:
                merged[hit.chunk_id] = hit
    return sorted(merged.values(), key=lambda item: (-item.score, item.chunk_id))


def rerank_hits(question: str, candidates: Sequence[SearchHit], *, limit: int) -> list[SearchHit]:
    question_terms = set(question_facets(question))
    scored: list[tuple[float, SearchHit]] = []
    for semantic_rank, hit in enumerate(candidates, start=1):
        text_terms = set(
            normalized_tokens(
                " ".join([hit.filename, hit.title, " ".join(hit.heading_path), hit.text])
            )
        )
        lexical_coverage = (
            len(question_terms & text_terms) / len(question_terms)
            if question_terms
            else 0.0
        )
        reciprocal_rank = 1.0 / semantic_rank
        neighbour_bonus = 0.03 if hit.metadata.get("retrieval_origin") == "neighbor" else 0.0
        rerank_score = (
            0.50 * max(0.0, min(1.0, hit.score))
            + 0.42 * lexical_coverage
            + 0.08 * reciprocal_rank
            + neighbour_bonus
        )
        metadata = dict(hit.metadata)
        metadata.update(
            semantic_score=hit.score,
            semantic_rank=semantic_rank,
            lexical_coverage=round(lexical_coverage, 6),
            rerank_score=round(rerank_score, 6),
        )
        scored.append(
            (
                rerank_score,
                hit.model_copy(update={"score": round(rerank_score, 8), "metadata": metadata}),
            )
        )
    scored.sort(key=lambda item: (-item[0], str(item[1].chunk_id)))

    # A small maximal-marginal-relevance penalty prevents near-duplicate chunks
    # from occupying the entire final context while still allowing several
    # distinct passages from one source document.
    selected: list[SearchHit] = []
    remaining = [item[1] for item in scored]
    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")
        for index, hit in enumerate(remaining):
            base = float(hit.metadata.get("rerank_score", hit.score))
            redundancy = max(
                (_token_jaccard(hit.text, chosen.text) for chosen in selected),
                default=0.0,
            )
            diversified = base - 0.12 * redundancy
            if diversified > best_score:
                best_index = index
                best_score = diversified
        chosen = remaining.pop(best_index)
        metadata = dict(chosen.metadata)
        metadata["diversified_score"] = round(best_score, 6)
        selected.append(chosen.model_copy(update={"metadata": metadata}))
    return selected


def retry_query(question: str, assessment: CoverageAssessment) -> str:
    missing = ", ".join(assessment.missing_facets[:8])
    return f"{question}\nEvidence focus: {missing}" if missing else question


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalized_tokens(left))
    right_tokens = set(normalized_tokens(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0
