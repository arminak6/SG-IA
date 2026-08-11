"""Application-facing orchestration for ingestion and grounded Q&A."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
from typing import Mapping, Sequence

from .agent import WikiAgent, build_ingestion_prompt
from .answer_fixes import AnswerFixReviewer
from .bedrock import BedrockConverseClient
from .confidence import ConfidenceEvaluation, ConfidenceEvaluator
from .config import BedrockSettings, load_settings
from .manager_actions import (
    ManagerActionContext,
    ManagerActionInterpreter,
    ManagerActionProposal,
    ManagerActionSession,
    ManagerActionStore,
)
from .embeddings import TitanEmbeddingClient
from .repository import RepositoryError, WikiRepository
from .search import HybridWikiSearch


logger = logging.getLogger(__name__)


class WikiService:
    """Coordinate deterministic repository work and sequential LLM operations."""

    MAX_CHAT_SESSIONS = 200

    def __init__(
        self,
        settings: BedrockSettings | None = None,
        *,
        repository: WikiRepository | None = None,
        bedrock: BedrockConverseClient | None = None,
        agent: WikiAgent | None = None,
        searcher: HybridWikiSearch | None = None,
        confidence_evaluator: ConfidenceEvaluator | None = None,
        correction_interpreter: ManagerActionInterpreter | None = None,
        correction_store: ManagerActionStore | None = None,
        answer_fix_reviewer: AnswerFixReviewer | None = None,
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
        self._correction_interpreter = correction_interpreter
        self._correction_store = correction_store
        self._answer_fix_reviewer = answer_fix_reviewer
        self._update_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self._sessions: OrderedDict[str, ManagerActionSession] = OrderedDict()

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

    @property
    def correction_interpreter(self) -> ManagerActionInterpreter:
        if self._correction_interpreter is None:
            self._correction_interpreter = ManagerActionInterpreter(self.bedrock)
        return self._correction_interpreter

    @property
    def correction_store(self) -> ManagerActionStore:
        if self._correction_store is None:
            self._correction_store = ManagerActionStore(self.repository)
        return self._correction_store

    @property
    def answer_fix_reviewer(self) -> AnswerFixReviewer:
        if self._answer_fix_reviewer is None:
            self._answer_fix_reviewer = AnswerFixReviewer(
                self.repository,
                self.bedrock,
                self.searcher,
            )
        return self._answer_fix_reviewer

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

    def ask(self, question: str, *, session_id: str | None = None) -> dict[str, object]:
        question = str(question).strip()
        started_at = time.perf_counter()
        normalized_session_id = str(session_id).strip() if session_id else None
        if normalized_session_id:
            if (
                self._session(normalized_session_id) is None
                and self.correction_interpreter.looks_like_action(question)
            ):
                self._set_session(
                    normalized_session_id,
                    ManagerActionSession(
                        context=ManagerActionContext(question="", answer="", citations=())
                    ),
                )
            correction_response = self._handle_correction_message(
                normalized_session_id,
                question,
                started_at=started_at,
            )
            if correction_response is not None:
                return correction_response

        response = self._answer_question(question, started_at=started_at)
        if normalized_session_id:
            self._remember_answer(normalized_session_id, question, response)
        return response

    def _answer_question(self, question: str, *, started_at: float) -> dict[str, object]:
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
        response = result.to_dict()
        confidence_score: float | None = None
        confidence: ConfidenceEvaluation | None = None
        verification_available = False
        try:
            confidence = self.confidence_evaluator.evaluate(question, result)
            verification_available = True
            confidence_score = confidence.score
            usage = response.setdefault("usage", {})
            if isinstance(usage, dict):
                for key, value in confidence.usage.items():
                    usage[key] = int(usage.get(key, 0)) + int(value)
        except Exception as exc:
            logger.warning("Confidence evaluation unavailable (%s)", type(exc).__name__)

        guardrail_reasons: tuple[str, ...]
        if result.status == "insufficient_knowledge":
            guardrail_reasons = ("answer_agent_abstained",)
        elif confidence is None:
            # A factual response cannot pass a semantic evidence gate when its
            # evidence verification is unavailable. Fail closed without a 503.
            guardrail_reasons = ("verification_unavailable",)
        else:
            guardrail_reasons = confidence.answer_guardrail_reasons()

        guardrail_applied = bool(guardrail_reasons)
        if guardrail_applied:
            language = confidence.response_language if confidence is not None else "other"
            response.update(
                {
                    "status": "insufficient_knowledge",
                    "answer": self._insufficient_knowledge_message(language),
                    "citations": [],
                }
            )
            if result.status == "answered" and confidence is not None:
                confidence_score = confidence.abstention_score

        debug = response.setdefault("debug", {})
        if isinstance(debug, dict):
            debug["guardrail"] = {
                "applied": guardrail_applied,
                "original_status": result.status,
                "verification_available": verification_available,
                "reasons": list(guardrail_reasons),
            }
        response.update(
            {
                "approach": "wiki",
                "model_id": self.settings.bedrock_model_id,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "confidence_score": confidence_score,
            }
        )

        final_citations = response.get("citations", [])
        logged_pages = [
            str(item.get("wiki_path"))
            for item in final_citations
            if isinstance(item, dict) and item.get("wiki_path")
        ] if isinstance(final_citations, list) else []
        try:
            self.repository.append_log(
                "query",
                question,
                status=str(response.get("status", result.status)),
                pages=logged_pages,
                detail=(
                    f"Evidence guardrail: {', '.join(guardrail_reasons)}"
                    if guardrail_applied
                    else None
                ),
            )
        except RepositoryError:
            # Logging is observability; it must not discard an already-grounded answer.
            pass
        return response

    def _handle_correction_message(
        self,
        session_id: str,
        message: str,
        *,
        started_at: float,
    ) -> dict[str, object] | None:
        session = self._session(session_id)
        if session is None:
            return None
        pending = session.pending

        if pending is not None and self.correction_interpreter.is_cancellation(message):
            self._set_session(session_id, replace(session, pending=None))
            return self._correction_response(
                status="manager_action_cancelled",
                answer=self._cancelled_message(pending.language),
                proposal=pending,
                state="cancelled",
                started_at=started_at,
            )

        if pending is not None and self.correction_interpreter.is_confirmation(message):
            if not pending.ready_for_confirmation:
                return self._proposal_response(pending, started_at=started_at)
            return self._apply_correction(
                session_id,
                session,
                pending,
                started_at=started_at,
            )

        should_interpret = pending is not None or self.correction_interpreter.looks_like_action(
            message
        )
        if not should_interpret:
            return None
        try:
            proposal = self.correction_interpreter.interpret(
                session.context,
                message,
                draft=pending,
            )
        except Exception as exc:
            logger.warning("Manager action interpretation unavailable (%s)", type(exc).__name__)
            return self._correction_response(
                status="manager_action_error",
                answer=self._correction_error_message(
                    pending.language if pending is not None else "other"
                ),
                proposal=pending,
                state="error",
                started_at=started_at,
            )
        if proposal is None:
            if pending is None:
                return None
            return self._proposal_response(pending, started_at=started_at)

        if proposal.action_type == "fix_answer" and not session.context.question:
            proposal = replace(
                proposal,
                needs_clarification=True,
                clarification_question=(
                    "Ask the Wiki question first, then submit Fix answer after the "
                    "incorrect response so the existing evidence can be reviewed."
                ),
            )

        self._set_session(session_id, replace(session, pending=proposal))
        return self._proposal_response(proposal, started_at=started_at)

    def _apply_correction(
        self,
        session_id: str,
        session: ManagerActionSession,
        proposal: ManagerActionProposal,
        *,
        started_at: float,
    ) -> dict[str, object]:
        if proposal.action_type == "fix_answer":
            return self._apply_answer_fix(
                session_id,
                session,
                proposal,
                started_at=started_at,
            )
        return self._apply_knowledge_action(
            session_id,
            session,
            proposal,
            started_at=started_at,
        )

    def _apply_knowledge_action(
        self,
        session_id: str,
        session: ManagerActionSession,
        proposal: ManagerActionProposal,
        *,
        started_at: float,
    ) -> dict[str, object]:
        try:
            source_path = self.correction_store.persist_knowledge(proposal, session.context)
            proposal = replace(proposal, source_path=source_path)
            self._set_session(session_id, replace(session, pending=proposal))
        except Exception as exc:
            logger.warning("Manager knowledge action persistence failed (%s)", type(exc).__name__)
            return self._correction_response(
                status="manager_action_failed",
                answer=self._correction_failure_message(proposal.language),
                proposal=proposal,
                state="failed",
                started_at=started_at,
            )

        update = self.update_wiki([source_path])
        processed = update.get("processed", [])
        skipped = update.get("skipped", [])
        failed = update.get("failed", [])
        already_ingested = (
            any(
                isinstance(item, Mapping)
                and str(item.get("source_path", "")).casefold()
                == source_path.casefold()
                and str(item.get("reason", "")).casefold() == "already ingested."
                for item in skipped
            )
            if isinstance(skipped, list)
            else False
        )
        if failed or (not processed and not already_ingested):
            return self._correction_response(
                status="manager_action_failed",
                answer=self._correction_ingestion_failure_message(proposal.language, source_path),
                proposal=proposal,
                state="failed",
                started_at=started_at,
            )

        pages_written: list[str] = []
        usage: dict[str, int] = dict(proposal.usage)
        for item in processed:
            if not isinstance(item, Mapping):
                continue
            for page in item.get("pages_written", []):
                if isinstance(page, str) and page not in pages_written:
                    pages_written.append(page)
            item_usage = item.get("usage", {})
            if isinstance(item_usage, Mapping):
                for key, value in item_usage.items():
                    if isinstance(value, int):
                        usage[str(key)] = usage.get(str(key), 0) + value

        citations: list[dict[str, object]] = []
        for page in sorted(pages_written, key=str.casefold):
            try:
                sources = self.repository.page_source_paths(page)
            except RepositoryError:
                continue
            citations.append({"wiki_path": page, "source_paths": sources})

        self._set_session(session_id, replace(session, pending=None))
        try:
            self.repository.append_log(
                "manager_action",
                proposal.subject,
                status="applied",
                pages=pages_written,
                detail=f"{proposal.action_id} | {proposal.action_type} | {source_path}",
            )
        except RepositoryError:
            pass
        return self._correction_response(
            status="manager_action_applied",
            answer=self._applied_message(proposal),
            proposal=proposal,
            state="applied",
            started_at=started_at,
            citations=citations,
            usage=usage,
        )

    def _apply_answer_fix(
        self,
        session_id: str,
        session: ManagerActionSession,
        proposal: ManagerActionProposal,
        *,
        started_at: float,
    ) -> dict[str, object]:
        try:
            plan = self.answer_fix_reviewer.prepare(session.context, proposal)
        except Exception as exc:
            logger.warning("Manager answer-fix review failed (%s)", type(exc).__name__)
            return self._correction_response(
                status="manager_action_failed",
                answer=self._answer_fix_failure_message(proposal.language),
                proposal=proposal,
                state="failed",
                started_at=started_at,
            )
        if not plan.supported:
            return self._correction_response(
                status="manager_action_failed",
                answer=self._answer_fix_unsupported_message(
                    proposal.language,
                    plan.explanation,
                ),
                proposal=proposal,
                state="failed",
                started_at=started_at,
                usage=plan.usage,
            )

        try:
            confidence = self.confidence_evaluator.evaluate(
                session.context.question,
                plan.answer_result(),
            )
        except Exception as exc:
            logger.warning("Answer-fix evidence verification failed (%s)", type(exc).__name__)
            return self._correction_response(
                status="manager_action_failed",
                answer=self._answer_fix_failure_message(proposal.language),
                proposal=proposal,
                state="failed",
                started_at=started_at,
                usage=plan.usage,
            )
        if confidence.answer_guardrail_reasons():
            return self._correction_response(
                status="manager_action_failed",
                answer=self._answer_fix_unsupported_message(
                    proposal.language,
                    "The corrected answer did not pass the existing evidence guardrail.",
                ),
                proposal=proposal,
                state="failed",
                started_at=started_at,
                usage=plan.usage,
            )

        evidence_pages = [citation.wiki_path for citation in plan.citations]
        try:
            with self._update_lock, self.repository.ingestion_lock():
                pages_updated = self.repository.apply_answer_fix_guidance(
                    action_id=proposal.action_id,
                    target_page=plan.target_page,
                    subject=proposal.subject,
                    guidance=plan.answer,
                    evidence_pages=evidence_pages,
                )
                if self.searcher.enabled:
                    try:
                        self.searcher.refresh()
                    except Exception as exc:
                        logger.warning(
                            "Semantic refresh deferred after answer fix (%s)",
                            type(exc).__name__,
                        )
        except Exception as exc:
            logger.warning("Answer-fix Wiki maintenance failed (%s)", type(exc).__name__)
            return self._correction_response(
                status="manager_action_failed",
                answer=self._answer_fix_failure_message(proposal.language),
                proposal=proposal,
                state="failed",
                started_at=started_at,
                usage=plan.usage,
            )

        result = {
            "corrected_answer": plan.answer,
            "failure_stage": plan.failure_stage,
            "explanation": plan.explanation,
            "evidence_pages": evidence_pages,
            "pages_updated": pages_updated,
            "confidence_score": confidence.score,
        }
        feedback_path: str | None = None
        try:
            feedback_path = self.correction_store.persist_answer_fix(
                proposal,
                session.context,
                result,
            )
        except Exception as exc:
            logger.warning("Answer-fix audit persistence failed (%s)", type(exc).__name__)

        proposal = replace(
            proposal,
            feedback_path=feedback_path,
            pages_updated=tuple(pages_updated),
        )
        self._set_session(session_id, replace(session, pending=None))
        citations = [citation.to_dict() for citation in plan.citations]
        usage = dict(proposal.usage)
        for values in (plan.usage, confidence.usage):
            for key, value in values.items():
                usage[key] = usage.get(key, 0) + int(value)
        try:
            self.repository.append_log(
                "answer_fix",
                session.context.question,
                status="applied",
                pages=pages_updated,
                detail=f"{proposal.action_id} | {plan.failure_stage}",
            )
        except RepositoryError:
            pass
        return self._correction_response(
            status="manager_action_applied",
            answer=self._answer_fix_applied_message(proposal.language, plan.answer),
            proposal=proposal,
            state="applied",
            started_at=started_at,
            citations=citations,
            usage=usage,
            confidence_score=confidence.score,
        )

    def _proposal_response(
        self,
        proposal: ManagerActionProposal,
        *,
        started_at: float,
    ) -> dict[str, object]:
        state = "needs_clarification" if not proposal.ready_for_confirmation else "proposed"
        status = (
            "manager_action_needs_clarification"
            if state == "needs_clarification"
            else "manager_action_proposed"
        )
        return self._correction_response(
            status=status,
            answer=self._proposal_message(proposal),
            proposal=proposal,
            state=state,
            started_at=started_at,
            usage=proposal.usage,
        )

    def _correction_response(
        self,
        *,
        status: str,
        answer: str,
        proposal: ManagerActionProposal | None,
        state: str,
        started_at: float,
        citations: Sequence[Mapping[str, object]] = (),
        usage: Mapping[str, int] | None = None,
        confidence_score: float | None = None,
    ) -> dict[str, object]:
        return {
            "approach": "wiki",
            "status": status,
            "answer": answer,
            "citations": [dict(item) for item in citations],
            "usage": {str(key): int(value) for key, value in (usage or {}).items()},
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "model_id": self.settings.bedrock_model_id,
            "confidence_score": confidence_score,
            "debug": {
                "pages_read": [],
                "search_queries": [],
                "search_modes": [],
                "retrieval_diagnostics": [],
                "guardrail": {
                    "applied": False,
                    "original_status": status,
                    "verification_available": False,
                    "reasons": [],
                },
            },
            "manager_action": proposal.to_dict(state=state) if proposal is not None else None,
            "correction": proposal.to_dict(state=state) if proposal is not None else None,
        }

    def _remember_answer(
        self,
        session_id: str,
        question: str,
        response: Mapping[str, object],
    ) -> None:
        raw_citations = response.get("citations", [])
        citations = tuple(
            dict(item) for item in raw_citations if isinstance(item, Mapping)
        ) if isinstance(raw_citations, list) else ()
        context = ManagerActionContext(
            question=question,
            answer=str(response.get("answer", "")),
            citations=citations,
        )
        self._set_session(session_id, ManagerActionSession(context=context))

    def _session(self, session_id: str) -> ManagerActionSession | None:
        with self._session_lock:
            value = self._sessions.get(session_id)
            if value is not None:
                self._sessions.move_to_end(session_id)
            return value

    def _set_session(self, session_id: str, session: ManagerActionSession) -> None:
        with self._session_lock:
            self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.MAX_CHAT_SESSIONS:
                self._sessions.popitem(last=False)

    @staticmethod
    def _proposal_message(proposal: ManagerActionProposal) -> str:
        effective = proposal.effective_period or "Not specified"
        labels = {
            "fix_answer": "Fix incorrect LLM answer",
            "update_knowledge": "Update existing knowledge",
            "add_knowledge": "Add new knowledge",
            "unclear": "Needs clarification",
        }
        action_label = labels.get(proposal.action_type, proposal.action_type)
        if proposal.language == "italian":
            lines = [
                "**Azione proposta dal manager — approvazione richiesta**",
                "",
                f"- Tipo di azione: {action_label}",
                f"- Modifica le fonti di conoscenza: {'Sì' if proposal.changes_knowledge else 'No'}",
                f"- Manutenzione della Wiki derivata: {'Sì' if proposal.wiki_maintenance else 'No'}",
                f"- Argomento: {proposal.subject or 'Da specificare'}",
                f"- Valore precedente: {proposal.previous_value or 'Non isolato'}",
                f"- Nuovo valore: {proposal.new_value or 'Da specificare'}",
                f"- Ambito: {proposal.scope or 'Da specificare'}",
                f"- Periodo di validità: {effective}",
            ]
            if proposal.ready_for_confirmation:
                lines.extend(
                    [
                        "",
                        "Nessuna azione è stata ancora applicata. Rispondi **Confermo** "
                        "per procedere, oppure **Annulla**.",
                    ]
                )
            else:
                lines.extend(["", proposal.clarification_question])
            return "\n".join(lines)
        lines = [
            "**Proposed manager action — approval required**",
            "",
            f"- Action type: {action_label}",
            f"- Changes source knowledge: {'Yes' if proposal.changes_knowledge else 'No'}",
            f"- Maintains derived Wiki pages: {'Yes' if proposal.wiki_maintenance else 'No'}",
            f"- Subject: {proposal.subject or 'Not specified'}",
            f"- Previous value: {proposal.previous_value or 'Not isolated'}",
            f"- New/corrected value: {proposal.new_value or 'Not specified'}",
            f"- Scope: {proposal.scope or 'Not specified'}",
            f"- Effective period: {effective}",
        ]
        if proposal.ready_for_confirmation:
            lines.extend(
                [
                    "",
                    "Nothing has been applied yet. Reply **Confirm** to proceed, or **Cancel**.",
                ]
            )
        else:
            lines.extend(["", proposal.clarification_question])
        return "\n".join(lines)

    @staticmethod
    def _applied_message(proposal: ManagerActionProposal) -> str:
        operation = (
            "updated existing knowledge"
            if proposal.action_type == "update_knowledge"
            else "added new knowledge"
        )
        if proposal.language == "italian":
            return (
                "Azione confermata, salvata come fonte immutabile e integrata nella Wiki.\n\n"
                f"**{proposal.subject}: {proposal.new_value}**"
            )
        return (
            f"Manager action confirmed: {operation}. The immutable source was integrated into the Wiki.\n\n"
            f"**{proposal.subject}: {proposal.new_value}**"
        )

    @staticmethod
    def _answer_fix_applied_message(language: str, answer: str) -> str:
        if language == "italian":
            return (
                "Correzione della risposta confermata e verificata sulle fonti Wiki. "
                "Nessuna nuova fonte di conoscenza è stata aggiunta; la rappresentazione "
                "Wiki esistente è stata chiarita e il caso è stato registrato per i test.\n\n"
                f"{answer}"
            )
        return (
            "Answer fix confirmed and verified against existing Wiki sources. No new "
            "source knowledge was added; the existing Wiki representation was clarified "
            "and the case was recorded for regression testing.\n\n"
            f"{answer}"
        )

    @staticmethod
    def _answer_fix_failure_message(language: str) -> str:
        if language == "italian":
            return (
                "La correzione della risposta non è stata applicata perché non è stato "
                "possibile verificarla e integrare in sicurezza la Wiki esistente."
            )
        return (
            "The answer fix was not applied because it could not be safely verified and "
            "integrated into the existing Wiki."
        )

    @staticmethod
    def _answer_fix_unsupported_message(language: str, detail: str) -> str:
        prefix = (
            "La correzione non è supportata completamente dalle pagine Wiki esistenti. "
            "Usa invece **Aggiorna conoscenza** o **Aggiungi conoscenza** se le fonti devono cambiare."
            if language == "italian"
            else "The correction is not fully supported by existing Wiki pages. Use "
            "**Update knowledge** or **Add knowledge** instead if the sources must change."
        )
        return f"{prefix}\n\n{detail}" if detail else prefix

    @staticmethod
    def _cancelled_message(language: str) -> str:
        if language == "italian":
            return "Azione annullata. La Wiki non è stata modificata."
        return "Manager action cancelled. The Wiki was not changed."

    @staticmethod
    def _correction_error_message(language: str) -> str:
        if language == "italian":
            return (
                "Non sono riuscito a strutturare l'azione. Specifica: Correggi risposta, "
                "Aggiorna conoscenza oppure Aggiungi conoscenza."
            )
        return (
            "I could not structure that manager action. Specify Fix answer, Update "
            "knowledge, or Add knowledge and provide the relevant details."
        )

    @staticmethod
    def _correction_failure_message(language: str) -> str:
        if language == "italian":
            return (
                "Non è stato possibile salvare la correzione; la Wiki non è "
                "stata aggiornata."
            )
        return "The correction could not be saved, so the Wiki was not updated."

    @staticmethod
    def _correction_ingestion_failure_message(language: str, source_path: str) -> str:
        if language == "italian":
            return (
                f"La fonte di correzione `{source_path}` è stata salvata, ma "
                "l'integrazione nella Wiki non è riuscita. Rimane disponibile "
                "come documento in attesa."
            )
        return (
            f"The correction source `{source_path}` was saved, but Wiki ingestion "
            "failed. It remains available as a pending document."
        )

    @staticmethod
    def _insufficient_knowledge_message(language: str) -> str:
        if language == "italian":
            return (
                "Posso rispondere solo in base ai documenti Wiki disponibili, che non "
                "contengono informazioni sufficienti per rispondere a questa domanda."
            )
        return (
            "I can only answer from the available Wiki documents, and they do not "
            "contain enough evidence to answer this question."
        )

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
