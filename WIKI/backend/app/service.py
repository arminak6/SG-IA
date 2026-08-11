"""Application-facing orchestration for ingestion and grounded Q&A."""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Sequence

from .agent import WikiAgent, build_ingestion_prompt
from .bedrock import BedrockConverseClient
from .confidence import ConfidenceEvaluator
from .config import BedrockSettings, load_settings
from .embeddings import TitanEmbeddingClient
from .repository import RepositoryError, WikiRepository
from .search import HybridWikiSearch


logger = logging.getLogger(__name__)


class WikiService:
    """Coordinate deterministic repository work and sequential LLM operations."""

    def __init__(
        self,
        settings: BedrockSettings | None = None,
        *,
        repository: WikiRepository | None = None,
        bedrock: BedrockConverseClient | None = None,
        agent: WikiAgent | None = None,
        searcher: HybridWikiSearch | None = None,
        confidence_evaluator: ConfidenceEvaluator | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.repository = repository or WikiRepository(
            self.settings.backend_root,
            max_source_bytes=self.settings.max_source_bytes,
            max_extracted_characters=self.settings.max_extracted_characters,
        )
        self._bedrock = bedrock
        self._agent = agent
        self._searcher = searcher
        self._confidence_evaluator = confidence_evaluator
        self._update_lock = threading.Lock()

    @property
    def bedrock(self) -> BedrockConverseClient:
        if self._bedrock is None:
            self._bedrock = BedrockConverseClient(self.settings)
        return self._bedrock

    @property
    def searcher(self) -> HybridWikiSearch:
        if self._searcher is None:
            embedder = (
                TitanEmbeddingClient(self.settings)
                if self.settings.semantic_search_enabled
                else None
            )
            self._searcher = HybridWikiSearch(self.repository, embedder)
        return self._searcher

    @property
    def agent(self) -> WikiAgent:
        if self._agent is None:
            self._agent = WikiAgent(
                self.repository,
                self.bedrock,
                max_steps=self.settings.max_agent_steps,
                searcher=self.searcher,
            )
        return self._agent

    @property
    def confidence_evaluator(self) -> ConfidenceEvaluator:
        if self._confidence_evaluator is None:
            self._confidence_evaluator = ConfidenceEvaluator(self.repository, self.bedrock)
        return self._confidence_evaluator

    def health(self) -> dict[str, object]:
        documents = self.repository.list_raw_documents()
        ingested = sum(document.is_ingested for document in documents)
        return {
            "status": "ok" if self.settings.is_configured else "configuration_required",
            "bedrock": {
                "configured": self.settings.is_configured,
                "model_id": self.settings.bedrock_model_id,
                "region_name": self.settings.region_name,
                "credentials_source": self.settings.credentials_source,
            },
            "documents": {
                "total": len(documents),
                "pending": len(documents) - ingested,
                "ingested": ingested,
            },
            "wiki_pages": self.repository.count_wiki_pages(),
        }

    def list_documents(self) -> list[dict[str, object]]:
        return [document.to_dict() for document in self.repository.list_raw_documents()]

    def list_wiki_pages(self) -> list[dict[str, object]]:
        return [page.to_dict() for page in self.repository.list_wiki_pages()]

    @staticmethod
    def _source_path(value: str) -> str:
        normalized = str(value).strip().replace("\\", "/")
        if not normalized.startswith("raw/"):
            normalized = f"raw/{normalized}"
        return WikiRepository.normalize_source_path(normalized)

    def update_wiki(self, relative_paths: Sequence[str] | None = None) -> dict[str, object]:
        """Sequentially ingest selected sources, or every currently pending source."""

        documents = self.repository.list_raw_documents()
        known = {document.source_path.casefold(): document for document in documents}
        requested: list[str] = []
        invalid: list[tuple[str, str]] = []

        if relative_paths is None:
            requested = [document.source_path for document in documents if not document.is_ingested]
        else:
            if isinstance(relative_paths, str):
                relative_paths = [relative_paths]
            seen: set[str] = set()
            for supplied in relative_paths:
                try:
                    source_path = self._source_path(supplied)
                except (RepositoryError, TypeError, ValueError) as exc:
                    invalid.append((str(supplied), str(exc)))
                    continue
                folded = source_path.casefold()
                if folded not in seen:
                    requested.append(source_path)
                    seen.add(folded)

        processed: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        failed: list[dict[str, str]] = [
            {"source_path": supplied, "error": error} for supplied, error in invalid
        ]

        with self._update_lock, self.repository.ingestion_lock():
            for source_path in requested:
                document = known.get(source_path.casefold())
                if document is None:
                    failed.append({"source_path": source_path, "error": "Raw source does not exist."})
                    continue
                # Re-check inside the lock in case another request completed it.
                if self.repository.is_ingested(document.source_path):
                    skipped.append(
                        {"source_path": document.source_path, "reason": "Already ingested."}
                    )
                    continue

                prompt = build_ingestion_prompt(document.source_path)
                try:
                    result = self.agent.ingest(prompt)
                    processed_item = result.to_dict()
                    try:
                        self.repository.append_log(
                            "ingest",
                            document.source_path,
                            status="success",
                            pages=result.pages_written,
                            detail=result.message,
                        )
                    except RepositoryError:
                        processed_item["warning"] = (
                            "Knowledge was committed, but the operation log could not be updated."
                        )
                    processed.append(processed_item)
                except Exception as exc:
                    error = str(exc) or type(exc).__name__
                    failed.append({"source_path": document.source_path, "error": error})
                    try:
                        self.repository.append_log(
                            "ingest",
                            document.source_path,
                            status="failed",
                            detail=error,
                        )
                    except RepositoryError:
                        # Preserve the primary ingestion failure in the response.
                        pass

        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "summary": {
                "requested": len(requested) + len(invalid),
                "processed": len(processed),
                "skipped": len(skipped),
                "failed": len(failed),
            },
        }

    def ask(self, question: str) -> dict[str, object]:
        question = str(question).strip()
        started_at = time.perf_counter()
        try:
            result = self.agent.answer(question)
        except Exception as exc:
            try:
                self.repository.append_log(
                    "query", question or "(empty question)", status="failed", detail=str(exc)
                )
            except RepositoryError:
                pass
            raise
        try:
            self.repository.append_log(
                "query",
                question,
                status=result.status,
                pages=(citation.wiki_path for citation in result.citations),
            )
        except RepositoryError:
            # Logging is observability; it must not discard an already grounded answer.
            pass
        response = result.to_dict()
        confidence_score: float | None = None
        try:
            confidence = self.confidence_evaluator.evaluate(question, result)
            confidence_score = confidence.score
            usage = response.setdefault("usage", {})
            if isinstance(usage, dict):
                for key, value in confidence.usage.items():
                    usage[key] = int(usage.get(key, 0)) + int(value)
        except Exception as exc:
            # Confidence is supplementary metadata. A verifier outage or malformed
            # verifier response must never discard an already-grounded answer.
            logger.warning("Confidence evaluation unavailable (%s)", type(exc).__name__)
        response.update(
            {
                "approach": "wiki",
                "model_id": self.settings.bedrock_model_id,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "confidence_score": confidence_score,
            }
        )
        return response

    def lint_wiki(self) -> dict[str, object]:
        return self.repository.lint_wiki()

    def repair_wiki_links(self, *, max_links: int = 12) -> dict[str, object]:
        """Run bounded semantic graph maintenance and apply safe cross-links."""

        with self._update_lock, self.repository.ingestion_lock():
            result = self.agent.repair_links(max_links=max_links)
            response = result.to_dict()
            try:
                self.repository.append_log(
                    "lint",
                    "semantic link repair",
                    status="success",
                    pages=result.pages_updated,
                    detail=f"Added {len(result.links_added)} semantic relationship(s).",
                )
            except RepositoryError:
                response["warning"] = (
                    "Cross-links were committed, but the operation log could not be updated."
                )
            return response


@lru_cache(maxsize=1)
def get_service() -> WikiService:
    """Return the process-wide service without making a live AWS request."""

    return WikiService(load_settings())


def reset_service_cache() -> None:
    """Clear the singleton for tests or an explicit configuration reload."""

    get_service.cache_clear()
