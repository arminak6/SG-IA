from __future__ import annotations

from functools import lru_cache

from .chunking import StructureAwareChunker
from .confidence import BedrockRagConfidenceEvaluator
from .config import Settings
from .embeddings import BedrockTitanEmbeddingProvider, build_boto3_session
from .extraction import CompositeExtractor, DoclingExtractor, TextDocumentExtractor
from .generation import BedrockGroundedAnswerGenerator
from .repository import LocalRepository
from .service import RagService
from .vector_store import QdrantVectorStore


def build_service(settings: Settings | None = None) -> RagService:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    session = build_boto3_session(
        region=settings.aws_region, credentials_file=settings.credentials_file
    )
    embeddings = BedrockTitanEmbeddingProvider(
        session=session,
        model_id=settings.embedding_model_id,
        dimension=settings.embedding_dimensions,
    )
    generator = BedrockGroundedAnswerGenerator(
        session=session,
        model_id=settings.generation_model_id,
        temperature=settings.generation_temperature,
        max_output_tokens=settings.generation_max_output_tokens,
        max_context_characters=settings.chat_max_context_characters,
    )
    confidence_evaluator = (
        BedrockRagConfidenceEvaluator(
            session=session,
            model_id=settings.confidence_model_id,
            max_output_tokens=settings.confidence_max_output_tokens,
            max_evidence_characters=settings.confidence_max_evidence_characters,
        )
        if settings.confidence_enabled
        else None
    )
    return RagService(
        settings=settings,
        repository=LocalRepository(settings),
        extractor=CompositeExtractor(
            DoclingExtractor(
                do_ocr=settings.docling_do_ocr,
                max_pages=settings.max_document_pages,
                max_characters=settings.max_extracted_characters,
            ),
            TextDocumentExtractor(
                max_characters=settings.max_extracted_characters
            ),
        ),
        chunker=StructureAwareChunker(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
        embeddings=embeddings,
        generator=generator,
        confidence_evaluator=confidence_evaluator,
        vector_store=QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            dimension=settings.embedding_dimensions,
        ),
    )


@lru_cache(maxsize=1)
def get_service() -> RagService:
    return build_service()
