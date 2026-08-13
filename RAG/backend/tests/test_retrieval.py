from __future__ import annotations

from app.models import SearchHit
from app.retrieval import assess_coverage, merge_hits, rerank_hits, retry_query


def hit(chunk_id: str, score: float, text: str, *, origin: str = "semantic") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="manual.pdf",
        title="Manual",
        score=score,
        text=text,
        heading_path=["Rules"],
        metadata={"ordinal": int(chunk_id[-1]), "retrieval_origin": origin},
    )


def test_reranker_can_promote_lexically_complete_evidence() -> None:
    candidates = [
        hit("chunk-1", 0.8, "Corporate graphic-design colour palette."),
        hit("chunk-2", 0.55, "Red means overdue, yellow means due soon, green means complete."),
    ]

    ranked = rerank_hits(
        "What do red, yellow and green mean?", candidates, limit=2
    )

    assert ranked[0].chunk_id == "chunk-2"
    assert ranked[0].metadata["semantic_score"] == 0.55
    assert ranked[0].metadata["lexical_coverage"] > ranked[1].metadata["lexical_coverage"]


def test_coverage_requires_numeric_facets_and_builds_retry_query() -> None:
    evidence = [hit("chunk-1", 0.8, "The milestones were 2010 and 2020.")]
    assessment = assess_coverage(
        "List the milestones in 2005, 2010 and 2020.",
        evidence,
        minimum_ratio=0.7,
    )

    assert assessment.sufficient is False
    assert "2005" in assessment.missing_facets
    assert "2005" in retry_query("List the milestones.", assessment)


def test_merge_hits_deduplicates_and_keeps_stronger_variant() -> None:
    merged = merge_hits(
        [hit("chunk-1", 0.4, "old")],
        [hit("chunk-1", 0.7, "better"), hit("chunk-2", 0.5, "other")],
    )

    assert len(merged) == 2
    assert next(item for item in merged if item.chunk_id == "chunk-1").text == "better"
