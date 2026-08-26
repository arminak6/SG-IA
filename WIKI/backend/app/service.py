"""Application-facing orchestration for ingestion and grounded Q&A."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
from typing import Mapping, Sequence

from .agent import AnswerSubmissionError, WikiAgent, build_ingestion_prompt
from .answer_fixes import AnswerFixReviewer
from .bedrock import BedrockConverseClient, BedrockError
from .confidence import ConfidenceEvaluation, ConfidenceEvaluator
from .config import BedrockSettings, load_settings
from .manager_actions import (
    ManagerActionContext,
    ManagerActionInterpreter,
    ManagerActionProposal,
    ManagerActionSession,
    ManagerActionStore,
    ManagerKnowledgeWrite,
)
from .embeddings import TitanEmbeddingClient
from .preference_interpreter import PreferenceDecision, PreferenceInterpreter
from .repository import RepositoryError, WikiRepository
from .search import HybridWikiSearch
from .user_memory import UserMemoryStore


logger = logging.getLogger(__name__)


class WikiService:
    """Coordinate deterministic repository work and sequential LLM operations."""

    MAX_CHAT_SESSIONS = 200
    MAX_ANSWER_ATTEMPTS = 4
    ANSWER_RETRY_BACKOFF_SECONDS = (0.25, 0.5, 1.0)
    WIKI_SCOPE_PATH_PATTERN = re.compile(
        r"(?:sources|concepts|entities|syntheses)/[^\s`'\"<>]+?\.md",
        flags=re.IGNORECASE,
    )
    MANAGER_UPDATE_ACTION_PATTERN = re.compile(
        r"^- Action type:\s*`update_knowledge`\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

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
        user_memory_store: UserMemoryStore | None = None,
        preference_interpreter: PreferenceInterpreter | None = None,
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
        self._preference_interpreter = preference_interpreter
        self.user_memory_store = user_memory_store or UserMemoryStore(
            self.settings.backend_root / "user_data"
        )
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

    @property
    def preference_interpreter(self) -> PreferenceInterpreter:
        if self._preference_interpreter is None:
            self._preference_interpreter = PreferenceInterpreter(self.bedrock)
        return self._preference_interpreter

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

    def get_user_profile(self, user_id: str) -> dict[str, object]:
        profile = self.user_memory_store.get_profile(user_id)
        if profile.updated_at is None:
            profile = self.user_memory_store.save_profile(profile.user_id, ())
        return profile.to_dict()

    def update_user_profile(
        self,
        user_id: str,
        preferences: Sequence[str],
    ) -> dict[str, object]:
        return self.user_memory_store.save_profile(user_id, preferences).to_dict()

    def reset_chat(self, user_id: str, session_id: str) -> dict[str, object]:
        normalized_user = self.user_memory_store.normalize_user_id(user_id)
        normalized_session = self.user_memory_store.normalize_session_id(session_id)
        cleared = self.user_memory_store.clear_session(
            normalized_user,
            normalized_session,
        )
        with self._session_lock:
            self._sessions.pop(normalized_session, None)
        return {
            "user_id": normalized_user,
            "session_id": normalized_session,
            "history_deleted": cleared,
        }

    @staticmethod
    def _source_path(value: str) -> str:
        normalized = str(value).strip().replace("\\", "/")
        if not normalized.startswith("raw/"):
            normalized = f"raw/{normalized}"
        return WikiRepository.normalize_source_path(normalized)

    def update_wiki(
        self,
        relative_paths: Sequence[str] | None = None,
        *,
        allow_manager_knowledge: bool = False,
    ) -> dict[str, object]:
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
                if (
                    self._is_manager_controlled_source(document.source_path)
                    and not allow_manager_knowledge
                ):
                    skipped.append(
                        {
                            "source_path": document.source_path,
                            "reason": (
                                "Manager knowledge sources require the confirmed "
                                "manager-action workflow."
                            ),
                        }
                    )
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

    def _is_manager_controlled_source(self, source_path: str) -> bool:
        normalized = self.repository.normalize_source_path(source_path)
        if normalized.casefold().startswith("raw/manager-knowledge/"):
            return True
        if not normalized.casefold().startswith("raw/manager-actions/"):
            return False
        try:
            content = self.repository.read_raw(normalized)
        except RepositoryError:
            return False
        return self.MANAGER_UPDATE_ACTION_PATTERN.search(content) is not None

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, object]:
        question = str(question).strip()
        started_at = time.perf_counter()
        normalized_session_id = str(session_id).strip() if session_id else None
        normalized_user_id = (
            self.user_memory_store.normalize_user_id(user_id) if user_id else None
        )
        conversation_history: tuple[dict[str, str], ...] = ()
        preferences: tuple[str, ...] = ()
        if normalized_user_id:
            preferences = self.user_memory_store.get_profile(
                normalized_user_id
            ).preferences
            if normalized_session_id:
                conversation_history = self.user_memory_store.read_session_context(
                    normalized_user_id,
                    normalized_session_id,
                )
        if normalized_session_id:
            if (
                self._session(normalized_session_id) is None
                and self.correction_interpreter.is_control_command(question)
            ):
                response = self._correction_response(
                    status="manager_action_not_pending",
                    answer=self._no_pending_action_message(),
                    proposal=None,
                    state="not_pending",
                    started_at=started_at,
                )
                return self._complete_chat(
                    normalized_user_id,
                    normalized_session_id,
                    question,
                    response,
                    include_in_context=False,
                )
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
                return self._complete_chat(
                    normalized_user_id,
                    normalized_session_id,
                    question,
                    correction_response,
                    include_in_context=False,
                )
            if self.correction_interpreter.looks_like_unmarked_action(question):
                response = self._correction_response(
                    status="manager_action_command_required",
                    answer=self._action_command_required_message(),
                    proposal=None,
                    state="command_required",
                    started_at=started_at,
                )
                return self._complete_chat(
                    normalized_user_id,
                    normalized_session_id,
                    question,
                    response,
                    include_in_context=False,
                )

        preference_decision: PreferenceDecision | None = None
        effective_preferences = preferences
        answer_question = question
        if normalized_user_id:
            preference_decision = self.preference_interpreter.interpret(
                question,
                current_preferences=preferences,
                conversation_history=conversation_history,
            )
            if preference_decision.requires_clarification:
                response = self._preference_response(
                    preference_decision,
                    "preference_clarification_required",
                    preferences,
                    started_at=started_at,
                )
                return self._complete_chat(
                    normalized_user_id,
                    normalized_session_id,
                    question,
                    response,
                    include_in_context=False,
                )
            if preference_decision.changes_persistent_preferences:
                profile = self.user_memory_store.apply_preference_changes(
                    normalized_user_id,
                    preferences_to_add=preference_decision.preferences_to_add,
                    preferences_to_remove=preference_decision.preferences_to_remove,
                    clear=preference_decision.operation == "clear",
                )
                effective_preferences = profile.preferences
                if preference_decision.is_preference_only:
                    status = (
                        "preferences_cleared"
                        if preference_decision.operation == "clear"
                        else "preference_saved"
                        if preference_decision.operation == "add"
                        else "preferences_updated"
                    )
                    response = self._preference_response(
                        preference_decision,
                        status,
                        effective_preferences,
                        started_at=started_at,
                    )
                    return self._complete_chat(
                        normalized_user_id,
                        normalized_session_id,
                        question,
                        response,
                        include_in_context=False,
                    )
                answer_question = preference_decision.remaining_question
            elif preference_decision.operation == "temporary":
                effective_preferences = tuple(
                    dict.fromkeys(
                        (*preferences, *preference_decision.preferences_to_add)
                    )
                )
                answer_question = preference_decision.remaining_question

        response = self._answer_question(
            answer_question,
            started_at=started_at,
            conversation_history=conversation_history,
            user_preferences=effective_preferences,
        )
        if preference_decision is not None:
            self._attach_preference_decision(response, preference_decision)
        if normalized_session_id:
            self._remember_answer(normalized_session_id, answer_question, response)
        return self._complete_chat(
            normalized_user_id,
            normalized_session_id,
            question,
            response,
            include_in_context=True,
        )

    def _answer_question(
        self,
        question: str,
        *,
        started_at: float,
        conversation_history: Sequence[Mapping[str, str]] = (),
        user_preferences: Sequence[str] = (),
    ) -> dict[str, object]:
        answer_attempts = 0
        answer_bedrock_failures = 0
        answer_submission_failures = 0
        for attempt in range(1, self.MAX_ANSWER_ATTEMPTS + 1):
            answer_attempts = attempt
            try:
                if conversation_history or user_preferences:
                    result = self.agent.answer(
                        question,
                        conversation_history=conversation_history,
                        user_preferences=user_preferences,
                    )
                else:
                    result = self.agent.answer(question)
                break
            except (BedrockError, AnswerSubmissionError) as exc:
                if isinstance(exc, BedrockError):
                    answer_bedrock_failures += 1
                else:
                    answer_submission_failures += 1
                if attempt < self.MAX_ANSWER_ATTEMPTS:
                    backoff_seconds = self.ANSWER_RETRY_BACKOFF_SECONDS[
                        min(attempt - 1, len(self.ANSWER_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    logger.warning(
                        "Wiki answer attempt %d/%d failed (%s); retrying the "
                        "read-only Q&A operation after %.2f seconds.",
                        attempt,
                        self.MAX_ANSWER_ATTEMPTS,
                        type(exc).__name__,
                        backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
                    continue
                try:
                    self.repository.append_log(
                        "query",
                        question or "(empty question)",
                        status="failed",
                        detail=str(exc),
                    )
                except RepositoryError:
                    pass
                raise
            except Exception as exc:
                try:
                    self.repository.append_log(
                        "query",
                        question or "(empty question)",
                        status="failed",
                        detail=str(exc),
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

        guardrail_applied = self.settings.answer_guardrail_enabled and bool(
            guardrail_reasons
        )
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
            debug["answer_attempts"] = answer_attempts
            debug["answer_retry_applied"] = answer_attempts > 1
            debug["answer_bedrock_failures"] = answer_bedrock_failures
            debug["answer_submission_failures"] = answer_submission_failures
            debug["history_messages_used"] = len(conversation_history)
            debug["user_preferences_used"] = len(user_preferences)
            debug["guardrail"] = {
                "enabled": self.settings.answer_guardrail_enabled,
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

    def _complete_chat(
        self,
        user_id: str | None,
        session_id: str | None,
        question: str,
        response: dict[str, object],
        *,
        include_in_context: bool,
    ) -> dict[str, object]:
        if not user_id or not session_id:
            return response
        history_saved = True
        try:
            self.user_memory_store.append_exchange(
                user_id,
                session_id,
                question,
                response,
                include_in_context=include_in_context,
            )
        except Exception as exc:
            history_saved = False
            logger.warning("Chat history could not be saved (%s)", type(exc).__name__)
        debug = response.setdefault("debug", {})
        if isinstance(debug, dict):
            debug["history_saved"] = history_saved
        return response

    def _preference_response(
        self,
        decision: PreferenceDecision,
        status: str,
        preferences: Sequence[str],
        *,
        started_at: float,
    ) -> dict[str, object]:
        italian = decision.language == "italian"
        preference_changed = (
            decision.changes_persistent_preferences
            and status != "preference_clarification_required"
        )
        if status == "preference_clarification_required":
            answer = decision.clarification_question
        elif status == "preferences_cleared":
            answer = (
                "Ho eliminato le tue preferenze salvate."
                if italian
                else "I cleared your saved preferences."
            )
        elif status == "preferences_updated":
            answer = (
                "Ho aggiornato le tue preferenze salvate e user\u00f2 quelle nuove nelle prossime chat."
                if italian
                else "I updated your saved preferences and will use the resolved preferences in future chats."
            )
        else:
            answer = (
                "Ho salvato questa preferenza e la user\u00f2 nelle prossime chat."
                if italian
                else "I saved this preference and will use it in future chats."
            )
        return {
            "approach": "wiki",
            "status": status,
            "answer": answer,
            "citations": [],
            "usage": dict(decision.usage),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "model_id": self.settings.bedrock_model_id,
            "confidence_score": None,
            "debug": {
                "pages_read": [],
                "search_queries": [],
                "search_modes": [],
                "retrieval_diagnostics": [],
                "answer_attempts": 1,
                "answer_retry_applied": False,
                "history_messages_used": 0,
                "user_preferences_used": len(preferences),
                "preference_detection_attempts": decision.attempts,
                "preference_operation": decision.operation,
                "preference_intent": decision.intent_kind,
                "preference_changed": preference_changed,
                "preference_clarification_required": decision.requires_clarification,
                "guardrail": {
                    "enabled": self.settings.answer_guardrail_enabled,
                    "applied": False,
                    "original_status": status,
                    "verification_available": False,
                    "reasons": [],
                },
            },
            "preference_changed": preference_changed,
            "preference_operation": decision.operation,
        }

    @staticmethod
    def _attach_preference_decision(
        response: dict[str, object],
        decision: PreferenceDecision,
    ) -> None:
        usage = response.setdefault("usage", {})
        if isinstance(usage, dict):
            for key, value in decision.usage.items():
                usage[str(key)] = int(usage.get(str(key), 0)) + int(value)
        debug = response.setdefault("debug", {})
        if isinstance(debug, dict):
            debug["preference_detection_attempts"] = decision.attempts
            debug["preference_operation"] = decision.operation
            debug["preference_intent"] = decision.intent_kind
            debug["preference_changed"] = decision.changes_persistent_preferences
            debug["preference_clarification_required"] = False
        response["preference_changed"] = decision.changes_persistent_preferences
        response["preference_operation"] = decision.operation

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

        explicit_action = self.correction_interpreter.looks_like_action(message)
        # Normal chat is always Q&A. Manager interpretation begins only with an
        # explicit command, or continues after such a command created a draft.
        if pending is None and not explicit_action:
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
                    "Ask the Wiki question first, then submit /fix after the "
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
        knowledge_write: ManagerKnowledgeWrite | None = None
        try:
            persisted = self.correction_store.persist_knowledge(proposal, session.context)
            if isinstance(persisted, ManagerKnowledgeWrite):
                knowledge_write = persisted
                source_path = persisted.source_path
            else:
                # Preserve compatibility with simple test/demonstration stores.
                source_path = str(persisted)
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

        if proposal.action_type == "update_knowledge":
            update = self._update_existing_knowledge(
                source_path,
                proposal=proposal,
                context=session.context,
            )
        else:
            update = self.update_wiki(
                [source_path], allow_manager_knowledge=True
            )
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
            if knowledge_write is not None:
                try:
                    self.correction_store.rollback_knowledge(knowledge_write)
                    proposal = replace(proposal, source_path=None)
                    self._set_session(
                        session_id,
                        replace(session, pending=proposal),
                    )
                except Exception as exc:
                    logger.warning(
                        "Manager knowledge rollback failed (%s)", type(exc).__name__
                    )
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

        proposal = replace(proposal, pages_updated=tuple(pages_written))
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

    def _update_existing_knowledge(
        self,
        source_path: str,
        *,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
    ) -> dict[str, object]:
        """Apply a manager update while making new Wiki paths impossible."""

        writable_pages = self._manager_update_targets(
            proposal,
            context,
            source_path=source_path,
        )
        processed: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []

        with self._update_lock, self.repository.ingestion_lock():
            if not self.repository.raw_exists(source_path):
                return {
                    "processed": [],
                    "skipped": [],
                    "failed": [
                        {"source_path": source_path, "error": "Raw source does not exist."}
                    ],
                }
            if self.repository.is_ingested(source_path):
                skipped.append({"source_path": source_path, "reason": "Already ingested."})
            else:
                try:
                    update_options: dict[str, object] = {"writable_pages": writable_pages}
                    if proposal.requires_exact_wording:
                        update_options["exact_approved_text"] = proposal.new_value
                    result = self.agent.update_existing_knowledge(source_path, **update_options)
                    processed_item = result.to_dict()
                    try:
                        self.repository.append_log(
                            "update_existing_knowledge",
                            source_path,
                            status="success",
                            pages=result.pages_written,
                            detail=result.message,
                        )
                    except RepositoryError:
                        processed_item["warning"] = (
                            "Knowledge was updated, but the operation log could not be updated."
                        )
                    processed.append(processed_item)
                except Exception as exc:
                    error = str(exc) or type(exc).__name__
                    failed.append({"source_path": source_path, "error": error})
                    try:
                        self.repository.append_log(
                            "update_existing_knowledge",
                            source_path,
                            status="failed",
                            detail=error,
                        )
                    except RepositoryError:
                        pass

        return {"processed": processed, "skipped": skipped, "failed": failed}

    def _manager_update_targets(
        self,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
        *,
        source_path: str,
    ) -> tuple[str, ...]:
        """Select existing canonical and source-summary pages for a stable source."""

        explicit = list(
            dict.fromkeys(
                self.repository.normalize_wiki_path(match, allow_system=False)
                for match in self.WIKI_SCOPE_PATH_PATTERN.findall(proposal.scope)
            )
        )

        cited: list[str] = []
        for citation in context.citations:
            value = citation.get("wiki_path") if isinstance(citation, Mapping) else None
            if not isinstance(value, str):
                continue
            try:
                path = self.repository.normalize_wiki_path(value, allow_system=False)
                self.repository.read_wiki_page(path)
            except RepositoryError:
                continue
            if path not in cited:
                cited.append(path)

        canonical = [path for path in cited if not path.casefold().startswith("sources/")]
        owned = list(self.repository.source_manifest_pages(source_path))
        provenance = self.repository.provenance_pages(source_path)
        selected = owned + explicit + (canonical or cited) + provenance
        return tuple(dict.fromkeys(selected))

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
            converted = self._convert_answer_fix_to_update(
                proposal,
                session.context,
                review_usage=plan.usage,
            )
            self._set_session(session_id, replace(session, pending=converted))
            return self._correction_response(
                status=(
                    "manager_action_proposed"
                    if converted.ready_for_confirmation
                    else "manager_action_needs_clarification"
                ),
                answer=self._answer_fix_converted_message(
                    converted,
                    plan.explanation,
                ),
                proposal=converted,
                state=(
                    "proposed"
                    if converted.ready_for_confirmation
                    else "needs_clarification"
                ),
                started_at=started_at,
                usage=converted.usage,
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

    @staticmethod
    def _convert_answer_fix_to_update(
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
        *,
        review_usage: Mapping[str, int],
    ) -> ManagerActionProposal:
        """Turn an ungrounded answer fix into a non-writing update proposal."""

        usage = dict(proposal.usage)
        for key, value in review_usage.items():
            usage[str(key)] = usage.get(str(key), 0) + int(value)

        scope = proposal.scope.strip()
        if not scope:
            wiki_paths = [
                str(citation.get("wiki_path", "")).strip()
                for citation in context.citations
                if isinstance(citation, Mapping)
            ]
            scope = ", ".join(dict.fromkeys(path for path in wiki_paths if path))

        previous_value = proposal.previous_value.strip() or context.answer.strip()
        converted = replace(
            proposal,
            action_type="update_knowledge",
            previous_value=previous_value,
            scope=scope,
            reason=(
                proposal.reason.strip()
                or "The manager correction is not supported by the maintained Wiki evidence."
            ),
            needs_clarification=False,
            clarification_question="",
            usage=usage,
        )
        if not converted.ready_for_confirmation:
            missing: list[str] = []
            if not converted.subject:
                missing.append("subject")
            if not converted.previous_value:
                missing.append("current value")
            if not converted.new_value:
                missing.append("new value")
            if not converted.scope:
                missing.append("scope")
            converted = replace(
                converted,
                needs_clarification=True,
                clarification_question=(
                    "Please provide the missing update details: " + ", ".join(missing) + "."
                ),
            )
        return converted

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
                    "enabled": self.settings.answer_guardrail_enabled,
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
            maintained_knowledge=self._maintained_manager_knowledge(citations),
        )
        self._set_session(session_id, ManagerActionSession(context=context))

    def _maintained_manager_knowledge(
        self,
        citations: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, str], ...]:
        """Load cited stable manager snapshots for a later merge preview."""

        prefix = "raw/manager-knowledge/"
        snapshots: list[dict[str, str]] = []
        seen: set[str] = set()
        for citation in citations:
            values = citation.get("source_paths", [])
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    source_path = self.repository.normalize_source_path(str(value))
                except RepositoryError:
                    continue
                folded = source_path.casefold()
                if not folded.startswith(prefix) or folded in seen:
                    continue
                try:
                    content = self.repository.read_raw(source_path)
                except RepositoryError:
                    continue
                marker = "## Current approved knowledge"
                current_value = content.split(marker, 1)[1].strip() if marker in content else content.strip()
                subject_match = re.search(
                    r"(?m)^#\s+Manager Knowledge:\s*(.+?)\s*$",
                    content,
                )
                scope_match = re.search(r"(?m)^- Scope:\s*(.+?)\s*$", content)
                period_match = re.search(
                    r"(?m)^- Effective period:\s*(.+?)\s*$",
                    content,
                )
                snapshots.append(
                    {
                        "source_path": source_path,
                        "current_value": current_value,
                        "subject": subject_match.group(1).strip() if subject_match else "",
                        "scope": scope_match.group(1).strip() if scope_match else "",
                        "effective_period": (
                            period_match.group(1).strip() if period_match else ""
                        ),
                    }
                )
                seen.add(folded)
        return tuple(snapshots)

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
            heading = (
                "**Azione proposta dal manager — approvazione richiesta**"
                if proposal.ready_for_confirmation
                else "**Azione del manager — servono altre informazioni**"
            )
            lines = [
                heading,
                "",
                f"- Tipo di azione: {action_label}",
                f"- Modifica le fonti di conoscenza: {'Sì' if proposal.changes_knowledge else 'No'}",
                f"- Gestione della Wiki derivata: {proposal.derived_wiki_operation}",
                f"- Argomento: {proposal.subject or 'Da specificare'}",
            ]
            if proposal.action_type != "add_knowledge":
                lines.append(
                    f"- Valore precedente: {proposal.previous_value or 'Da specificare'}"
                )
            lines.extend(
                [
                    f"- Conoscenza completa proposta: {proposal.new_value or 'Da specificare'}",
                    f"- Ambito: {proposal.scope or 'Da specificare'}",
                    f"- Periodo di validità: {effective}",
                ]
            )
            if proposal.merge_warnings:
                lines.append(
                    "- Revisione fusione: rimosse inferenze non supportate: "
                    + "; ".join(proposal.merge_warnings)
                )
            if proposal.ready_for_confirmation:
                lines.extend(
                    [
                        "",
                        "Nessuna azione è stata ancora applicata. Rispondi **/confermo** "
                        "per procedere, oppure **/annulla**.",
                    ]
                )
            else:
                lines.extend(["", proposal.clarification_question])
            return "\n".join(lines)
        heading = (
            "**Proposed manager action — approval required**"
            if proposal.ready_for_confirmation
            else "**Manager action — more information required**"
        )
        lines = [
            heading,
            "",
            f"- Action type: {action_label}",
            f"- Changes source knowledge: {'Yes' if proposal.changes_knowledge else 'No'}",
            f"- Derived Wiki handling: {proposal.derived_wiki_operation}",
            f"- Subject: {proposal.subject or 'Not specified'}",
        ]
        if proposal.action_type != "add_knowledge":
            lines.append(
                f"- Previous value: {proposal.previous_value or 'Not specified'}"
            )
        lines.extend(
            [
                f"- Proposed complete knowledge: {proposal.new_value or 'Not specified'}",
                f"- Scope: {proposal.scope or 'Not specified'}",
                f"- Effective period: {effective}",
            ]
        )
        if proposal.merge_warnings:
            lines.append(
                "- Merge review: removed unsupported inferred wording: "
                + "; ".join(proposal.merge_warnings)
            )
        if proposal.ready_for_confirmation:
            lines.extend(
                [
                    "",
                    "Nothing has been applied yet. Reply **/confirm** to proceed, or **/cancel**.",
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
                "Azione confermata: la fonte stabile è stata aggiornata e integrata "
                "nella Wiki.\n\n"
                f"**{proposal.subject}: {proposal.new_value}**"
            )
        return (
            f"Manager action confirmed: {operation}. The stable source was integrated "
            "into the Wiki.\n\n"
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

    @classmethod
    def _answer_fix_converted_message(
        cls,
        proposal: ManagerActionProposal,
        detail: str,
    ) -> str:
        if proposal.language == "italian":
            prefix = (
                "La correzione non è supportata dalle conoscenze Wiki esistenti, quindi "
                "è stata convertita in una proposta di **Aggiornamento conoscenza**. "
                "Nessuna modifica è stata ancora applicata ed è necessaria una nuova conferma."
            )
        else:
            prefix = (
                "The correction is not supported by the existing Wiki knowledge, so it "
                "has been converted to an **Update knowledge** proposal. Nothing has been "
                "changed yet, and a new confirmation is required."
            )
        explanation = f"\n\n{detail}" if detail else ""
        return f"{prefix}{explanation}\n\n{cls._proposal_message(proposal)}"

    @staticmethod
    def _cancelled_message(language: str) -> str:
        if language == "italian":
            return "Azione annullata. La Wiki non è stata modificata."
        return "Manager action cancelled. The Wiki was not changed."

    @staticmethod
    def _correction_error_message(language: str) -> str:
        if language == "italian":
            return (
                "Non sono riuscito a strutturare l'azione. Inizia con /fix, /update "
                "oppure /add."
            )
        return (
            "I could not structure that manager action. Start with /fix, /update, "
            "or /add and provide the relevant details."
        )

    @staticmethod
    def _no_pending_action_message() -> str:
        return (
            "There is no pending manager action to confirm or cancel. Start the "
            "action again with Fix answer, Update knowledge, or Add knowledge, "
            "then review its preview."
        )

    @staticmethod
    def _action_command_required_message() -> str:
        return (
            "This looks like a knowledge replacement, but normal chat is Q&A only. "
            "Click **Update knowledge** and enter the short change in the manager "
            "form. The interface adds `/update` automatically."
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
                f"L'integrazione della modifica da `{source_path}` non è riuscita. "
                "La fonte stabile e la Wiki sono state ripristinate; la proposta "
                "rimane in attesa e può essere annullata o riprovata."
            )
        return (
            f"Wiki integration for `{source_path}` failed. The stable source and Wiki "
            "were restored; the proposal remains pending and can be cancelled or retried."
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
