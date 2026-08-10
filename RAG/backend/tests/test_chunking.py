from app.chunking import StructureAwareChunker
from app.models import DocumentElement, ElementType


def element(
    index: int,
    text: str,
    kind: ElementType = ElementType.TEXT,
    heading: list[str] | None = None,
    page: int | None = 1,
) -> DocumentElement:
    return DocumentElement(
        element_id=f"e-{index}",
        element_type=kind,
        text=text,
        heading_path=heading or [],
        page_number=page,
    )


def test_chunks_do_not_cross_section_boundaries() -> None:
    chunks = StructureAwareChunker(max_tokens=50, overlap_tokens=5).chunk(
        document_id="doc",
        source_hash="abc",
        filename="policy.md",
        title="Policy",
        elements=[
            element(1, "First section", ElementType.HEADING, ["First"]),
            element(2, "Alpha details", heading=["First"]),
            element(3, "Second section", ElementType.HEADING, ["Second"], 2),
            element(4, "Beta details", heading=["Second"], page=2),
        ],
    )

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["First"]
    assert chunks[1].heading_path == ["Second"]
    assert chunks[1].page_numbers == [2]


def test_table_is_an_atomic_chunk_with_provenance() -> None:
    chunks = StructureAwareChunker(max_tokens=50, overlap_tokens=5).chunk(
        document_id="doc",
        source_hash="abc",
        filename="rules.pdf",
        title="Rules",
        elements=[
            element(1, "Introduction", heading=["Scope"]),
            element(
                2,
                "| Rule | Value |\n|---|---|\n| A | 3 |",
                ElementType.TABLE,
                ["Scope"],
                4,
            ),
        ],
    )

    assert len(chunks) == 2
    assert chunks[1].content_types == [ElementType.TABLE]
    assert chunks[1].page_numbers == [4]
    assert "Document: Rules" in chunks[1].embedding_text


def test_oversized_element_uses_overlap() -> None:
    text = " ".join(f"word{index}" for index in range(120))
    chunks = StructureAwareChunker(max_tokens=50, overlap_tokens=10).chunk(
        document_id="doc",
        source_hash="abc",
        filename="long.txt",
        title="Long",
        elements=[element(1, text)],
    )

    assert len(chunks) == 3
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert len({chunk.chunk_id for chunk in chunks}) == 3

