from __future__ import annotations

import hashlib
import re
import uuid

from .models import DocumentElement, ElementType, RagChunk

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ATOMIC_TYPES = {
    ElementType.TABLE,
    ElementType.PICTURE,
    ElementType.FORMULA,
    ElementType.CODE,
}


def estimate_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


class StructureAwareChunker:
    def __init__(self, max_tokens: int = 600, overlap_tokens: int = 100):
        if max_tokens < 50:
            raise ValueError("max_tokens must be at least 50")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be below max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        *,
        document_id: str,
        source_hash: str,
        filename: str,
        title: str,
        elements: list[DocumentElement],
    ) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        group: list[DocumentElement] = []
        group_heading: list[str] | None = None

        def flush() -> None:
            nonlocal group
            if group:
                chunks.extend(
                    self._emit_group(
                        document_id=document_id,
                        source_hash=source_hash,
                        filename=filename,
                        title=title,
                        elements=group,
                        ordinal_start=len(chunks),
                    )
                )
                group = []

        for element in elements:
            if not element.text.strip():
                continue
            if element.element_type in _ATOMIC_TYPES:
                flush()
                chunks.extend(
                    self._emit_group(
                        document_id=document_id,
                        source_hash=source_hash,
                        filename=filename,
                        title=title,
                        elements=[element],
                        ordinal_start=len(chunks),
                    )
                )
                group_heading = None
                continue

            heading = element.heading_path
            proposed = group + [element]
            if group and (
                heading != group_heading
                or estimate_tokens(self._join_text(proposed)) > self.max_tokens
            ):
                flush()
            if not group:
                group_heading = heading
            group.append(element)

        flush()
        return chunks

    def _emit_group(
        self,
        *,
        document_id: str,
        source_hash: str,
        filename: str,
        title: str,
        elements: list[DocumentElement],
        ordinal_start: int,
    ) -> list[RagChunk]:
        text = self._join_text(elements)
        words = text.split()
        if estimate_tokens(text) <= self.max_tokens:
            parts = [text]
        else:
            # Word windows are used only for an indivisible oversized unit.
            window = max(1, self.max_tokens)
            step = max(1, window - self.overlap_tokens)
            parts = [" ".join(words[index : index + window]) for index in range(0, len(words), step)]
            parts = [part for part in parts if part]

        heading_path = elements[0].heading_path if elements else []
        page_numbers = sorted(
            {element.page_number for element in elements if element.page_number is not None}
        )
        content_types = sorted(
            {element.element_type for element in elements}, key=lambda item: item.value
        )
        element_ids = [element.element_id for element in elements]
        emitted: list[RagChunk] = []
        for offset, part in enumerate(parts):
            ordinal = ordinal_start + offset
            context = [f"Document: {title}", f"Source file: {filename}"]
            if heading_path:
                context.append(f"Section: {' > '.join(heading_path)}")
            embedding_text = "\n".join(context) + "\n\n" + part
            digest = hashlib.sha256(
                f"{document_id}:{ordinal}:{part}".encode()
            ).hexdigest()
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, digest))
            emitted.append(
                RagChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_hash=source_hash,
                    filename=filename,
                    title=title,
                    ordinal=ordinal,
                    text=part,
                    embedding_text=embedding_text,
                    token_count=estimate_tokens(part),
                    element_ids=element_ids,
                    page_numbers=page_numbers,
                    heading_path=heading_path,
                    content_types=content_types,
                )
            )
        return emitted

    @staticmethod
    def _join_text(elements: list[DocumentElement]) -> str:
        return "\n\n".join(element.text.strip() for element in elements if element.text.strip())
