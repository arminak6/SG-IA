"""Approved manager actions for the WIKI proof of concept."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping

from .bedrock import BedrockConverseClient, ConverseTurn
from .repository import RepositoryError, WikiRepository


ACTION_TYPES = frozenset({"fix_answer", "update_knowledge", "add_knowledge"})


class ManagerActionError(RuntimeError):
    """Raised when a manager action cannot be interpreted or persisted."""


@dataclass(frozen=True)
class ManagerActionContext:
    question: str
    answer: str
    citations: tuple[dict[str, object], ...]
    maintained_knowledge: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [dict(item) for item in self.citations],
            "maintained_knowledge": [dict(item) for item in self.maintained_knowledge],
        }


@dataclass(frozen=True)
class ManagerActionProposal:
    action_id: str
    action_type: str
    subject: str
    previous_value: str
    new_value: str
    scope: str
    effective_period: str
    reason: str
    needs_clarification: bool
    clarification_question: str
    language: str
    usage: dict[str, int]
    manager_input: str = ""
    source_path: str | None = None
    feedback_path: str | None = None
    pages_updated: tuple[str, ...] = ()
    merge_warnings: tuple[str, ...] = ()

    @property
    def changes_knowledge(self) -> bool:
        return self.action_type in {"update_knowledge", "add_knowledge"}

    @property
    def wiki_maintenance(self) -> bool:
        return self.action_type == "fix_answer"

    @property
    def derived_wiki_operation(self) -> str:
        return {
            "fix_answer": "Repairs an existing evidence page",
            "update_knowledge": "Rewrites existing cited pages",
            "add_knowledge": "Creates or updates derived pages",
        }.get(self.action_type, "Not determined")

    @property
    def requires_exact_wording(self) -> bool:
        """Whether the manager explicitly requested verbatim preservation."""

        normalized = re.sub(r"\s+", " ", self.manager_input.casefold())
        return any(
            marker in normalized
            for marker in (
                "exact wording",
                "exactly:",
                "verbatim",
                "word for word",
                "esattamente:",
                "testo esatto",
                "parola per parola",
            )
        )

    @property
    def ready_for_confirmation(self) -> bool:
        if self.needs_clarification or self.action_type not in ACTION_TYPES:
            return False
        if not self.subject or not self.new_value:
            return False
        if self.action_type == "update_knowledge" and not self.previous_value:
            return False
        return True

    def to_dict(self, *, state: str) -> dict[str, object]:
        return {
            "state": state,
            "action_id": self.action_id,
            "correction_id": self.action_id,
            "action_type": self.action_type,
            "changes_knowledge": self.changes_knowledge,
            "wiki_maintenance": self.wiki_maintenance,
            "derived_wiki_operation": self.derived_wiki_operation,
            "subject": self.subject,
            "previous_value": self.previous_value,
            "corrected_value": self.new_value,
            "proposed_knowledge": self.new_value,
            "manager_input": self.manager_input,
            "scope": self.scope,
            "effective_period": self.effective_period,
            "source_path": self.source_path,
            "feedback_path": self.feedback_path,
            "pages_updated": list(self.pages_updated),
            "merge_warnings": list(self.merge_warnings),
        }


@dataclass(frozen=True)
class ManagerActionSession:
    context: ManagerActionContext
    pending: ManagerActionProposal | None = None


@dataclass(frozen=True)
class ManagerKnowledgeWrite:
    """One reversible write to a stable manager-maintained source."""

    source_path: str
    previous_content: str | None


MANAGER_ACTION_TOOL = {
    "toolSpec": {
        "name": "submit_manager_action",
        "description": "Structure an explicitly requested trusted-manager action.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_manager_action": {"type": "boolean"},
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "fix_answer",
                            "update_knowledge",
                            "add_knowledge",
                            "unclear",
                        ],
                    },
                    "subject": {"type": "string"},
                    "previous_value": {"type": "string"},
                    "new_value": {"type": "string"},
                    "scope": {"type": "string"},
                    "effective_period": {"type": "string"},
                    "reason": {"type": "string"},
                    "needs_clarification": {"type": "boolean"},
                    "clarification_question": {"type": "string"},
                    "language": {
                        "type": "string",
                        "enum": ["english", "italian", "other"],
                    },
                },
                "required": [
                    "is_manager_action",
                    "action_type",
                    "subject",
                    "previous_value",
                    "new_value",
                    "scope",
                    "effective_period",
                    "reason",
                    "needs_clarification",
                    "clarification_question",
                    "language",
                ],
                "additionalProperties": False,
            }
        },
    }
}


MANAGER_MERGE_REVIEW_TOOL = {
    "toolSpec": {
        "name": "review_manager_merge",
        "description": "Verify that merged manager knowledge contains no unsupported additions.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "corrected_value": {"type": "string"},
                    "unsupported_additions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "explanation": {"type": "string"},
                },
                "required": [
                    "valid",
                    "corrected_value",
                    "unsupported_additions",
                    "explanation",
                ],
                "additionalProperties": False,
            }
        },
    }
}


MANAGER_MERGE_REVIEW_PROMPT = """You verify a proposed trusted-manager knowledge merge.

The maintained_knowledge snapshot is the complete prior approved source. The manager_input contains the
new instruction. Every material claim in proposed_knowledge must be entailed by one of those inputs.
Normal grammatical paraphrase is allowed, but a new characterization, purpose, date, quantity, actor,
scope, recurrence, consequence, or certainty is unsupported unless an input states or necessarily entails
it. For example, an "email" must not become a "reminder email" unless an input calls it a reminder.

The manager_input may replace, qualify, or append to the prior snapshot. Preserve prior facts that are not
contradicted, and apply the manager's explicit change. If the proposal is fully supported, return valid=true
and the proposal unchanged in corrected_value. Otherwise return valid=false, list concise unsupported
additions, and provide a complete corrected_value with only those additions removed. Never add new facts.
Call review_manager_merge exactly once.
"""


MANAGER_ACTION_PROMPT = """You structure an explicitly requested action from a trusted manager in a Wiki proof of concept.

The backend accepts a new manager action only when the message starts with /fix, /update, or /add.
The application-supplied required_action_type is authoritative; never change it based on the prose that
follows the command. Use only the supplied previous interaction, draft, and manager message:
- fix_answer: the Wiki already contains the correct information, but the previous LLM answer retrieved,
  interpreted, ordered, or expressed it incorrectly. This adds no source knowledge.
- update_knowledge: maintained knowledge is outdated or wrong and the manager supplies a replacement.
- add_knowledge: the manager supplies genuinely new information absent from maintained knowledge.
- unclear: required details for the selected action are missing or ambiguous.

Combine all supplied changes into one complete new_value. For update_knowledge, new_value is the complete
current approved statement after applying the manager's change to the previous approved knowledge; it is
never just the incremental instruction or raw manager wording. Preserve still-valid facts, replace facts
the manager explicitly corrects, and retain uncertainty, modality, and confirmation conditions exactly.
Do not turn relative timing into a calculated calendar date unless the manager supplied that date. Use
add_knowledge only for a genuinely new standalone subject or record, not for an extra detail that belongs
to the existing subject. While refining an existing draft, preserve its action type and treat the new
message only as clarification.

Managers may write one short human sentence. When it is unambiguous, reuse the subject, previous value,
scope, cited context, and maintained_knowledge snapshot from the previous interaction instead of requiring
the manager to restate them. A proposed update new_value may contain still-valid facts from the previous
approved answer or maintained_knowledge snapshot plus facts the
manager explicitly supplied, with minor grammatical normalization. It must not add any other claim. Never
infer extra historical occurrences, dates, applicability, or consequences from the previous answer or
from a recurring rule. Previous interaction context may fill subject, previous_value, scope, and the
still-valid baseline that is explicitly visible in the complete merged preview.
A recurring statement such as "every year" or "ogni anno" is complete without a calendar year: preserve
the recurrence in new_value and record it as the effective_period. A complete calendar date supplies its
own effective year. Ask only for information that is genuinely missing; do not demand formal field labels
or polished wording.

Call submit_manager_action exactly once with is_manager_action=true. Request clarification when required
details for the selected action are ambiguous. Preserve useful draft fields and prior citation scope. For
time-dependent knowledge changes, require the effective period. If a calendar date is incomplete or
invalid without a year, request the missing detail instead of inventing it. Never invent domain details.
"""


class ManagerActionInterpreter:
    """Interpret manager messages into one of three explicit action types."""

    _ACTION_COMMAND = re.compile(
        r"^\s*/(?P<command>fix|update|add)(?=\s|$)", re.IGNORECASE
    )
    _CONTROL_COMMAND = re.compile(
        r"^\s*/(?:confirm|approve|cancel|confermo|approva|annulla)\s*$",
        re.IGNORECASE,
    )
    _UNMARKED_REPLACEMENT = re.compile(
        r"^\s*(?:for\b.{0,300}?,\s*)?replace\s+.+?\s+with\s+.+",
        re.IGNORECASE | re.DOTALL,
    )
    _RECURRING_PERIOD = re.compile(
        r"\b(?:every\s+(?:day|week|month|year)|each\s+(?:day|week|month|year)|"
        r"daily|weekly|monthly|annual(?:ly)?|yearly|ogni\s+(?:giorno|settimana|mese|anno)|"
        r"quotidianamente|settimanalmente|mensilmente|annuale|annualmente)\b",
        re.IGNORECASE,
    )
    _EFFECTIVE_PERIOD_QUESTION = re.compile(
        r"(?:effective\s+period|which\s+year|what\s+year|periodo\s+di\s+validit|"
        r"quale\s+anno)",
        re.IGNORECASE,
    )
    _ACTION_CHOICE_QUESTION = re.compile(
        r"(?:whether|se).{0,120}(?:fix|answer|update|existing knowledge|add|new knowledge|"
        r"corregg|risposta|aggiorn|conoscenza|aggiung)",
        re.IGNORECASE | re.DOTALL,
    )
    _SCOPE_QUESTION = re.compile(
        r"(?:scope|ambito|which (?:document|page|source|subject)|quale (?:documento|pagina|fonte))",
        re.IGNORECASE,
    )
    _ACTION_TYPES_BY_COMMAND = {
        "fix": "fix_answer",
        "update": "update_knowledge",
        "add": "add_knowledge",
    }
    _CONFIRMATIONS = frozenset({"confirm", "approve", "confermo", "approva"})
    _CANCELLATIONS = frozenset({"cancel", "annulla"})

    def __init__(self, bedrock: BedrockConverseClient) -> None:
        self.bedrock = bedrock

    @classmethod
    def looks_like_action(cls, message: str) -> bool:
        return cls.explicit_action_type(message) is not None

    @classmethod
    def explicit_action_type(cls, message: str) -> str | None:
        match = cls._ACTION_COMMAND.search(message)
        if match is None:
            return None
        return cls._ACTION_TYPES_BY_COMMAND[match.group("command").casefold()]

    looks_like_correction = looks_like_action

    @staticmethod
    def _command(message: str) -> str:
        return re.sub(r"[^\wàèéìòù]+", " ", message.casefold()).strip()

    @classmethod
    def is_confirmation(cls, message: str) -> bool:
        return cls._command(message) in cls._CONFIRMATIONS

    @classmethod
    def is_cancellation(cls, message: str) -> bool:
        return cls._command(message) in cls._CANCELLATIONS

    @classmethod
    def is_control_command(cls, message: str) -> bool:
        return cls._CONTROL_COMMAND.fullmatch(message) is not None

    @classmethod
    def looks_like_unmarked_action(cls, message: str) -> bool:
        return "?" not in message and cls._UNMARKED_REPLACEMENT.match(message) is not None

    def interpret(
        self,
        context: ManagerActionContext,
        manager_message: str,
        *,
        draft: ManagerActionProposal | None = None,
    ) -> ManagerActionProposal | None:
        command_action_type = self.explicit_action_type(manager_message)
        required_action_type = (
            draft.action_type if draft is not None else command_action_type
        )
        if required_action_type is None:
            return None
        action_details = self._action_details(
            manager_message,
            required_action_type=required_action_type,
            draft=draft,
        )
        payload = {
            "previous_interaction": context.to_dict(),
            "existing_draft": self._draft_payload(draft),
            "manager_message": action_details,
            "required_action_type": required_action_type,
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "<manager_action_data>\n"
                            f"{json.dumps(payload, ensure_ascii=False)}\n"
                            "</manager_action_data>"
                        )
                    }
                ],
            }
        ]
        usage: dict[str, int] = {}
        for attempt in range(2):
            turn = self.bedrock.converse(
                messages=messages,
                system_prompt=MANAGER_ACTION_PROMPT,
                tools=[MANAGER_ACTION_TOOL],
                max_tokens=1_000,
                temperature=0,
            )
            self._add_usage(usage, turn)
            submissions = self._submissions(turn)
            if len(submissions) == 1:
                try:
                    proposal = self._proposal(
                        submissions[0],
                        draft=draft,
                        usage=usage,
                        required_action_type=required_action_type,
                    )
                    proposal = self._preserve_explicit_knowledge_value(
                        proposal,
                        action_details=action_details,
                        draft=draft,
                    )
                    proposal = self._complete_explicit_proposal(proposal, context=context)
                    return self._review_knowledge_merge(proposal, context=context)
                except ManagerActionError:
                    if attempt == 1:
                        break
            if attempt == 0:
                messages.extend(
                    [
                        turn.message,
                        {
                            "role": "user",
                            "content": [{"text": "Call submit_manager_action exactly once now."}],
                        },
                    ]
                )
        return self._clarification_proposal(
            required_action_type,
            draft=draft,
            usage=usage,
        )

    @classmethod
    def _preserve_explicit_knowledge_value(
        cls,
        proposal: ManagerActionProposal,
        *,
        action_details: str,
        draft: ManagerActionProposal | None,
    ) -> ManagerActionProposal:
        """Keep an explicit manager value authoritative across LLM structuring."""

        if proposal.action_type not in {"update_knowledge", "add_knowledge"}:
            return proposal

        manager_input = action_details.strip()
        if draft is not None and draft.manager_input and manager_input:
            manager_input = f"{draft.manager_input}\n{manager_input}"
        elif draft is not None and not manager_input:
            manager_input = draft.manager_input

        # The structuring model produces the complete merged value used in the
        # preview. Fall back to the manager's exact text only if the model
        # omitted it entirely; never overwrite a valid merged proposal with an
        # incremental instruction.
        proposal = replace(proposal, manager_input=manager_input)
        if not proposal.new_value and manager_input:
            proposal = replace(proposal, new_value=manager_input)

        recurrence = cls._RECURRING_PERIOD.search(proposal.new_value)
        if recurrence is not None and not proposal.effective_period:
            proposal = replace(proposal, effective_period=recurrence.group(0))

        clarification_only_requests_period = bool(
            proposal.clarification_question
            and cls._EFFECTIVE_PERIOD_QUESTION.search(proposal.clarification_question)
        )
        if recurrence is not None and clarification_only_requests_period:
            proposal = replace(
                proposal,
                needs_clarification=False,
                clarification_question="",
            )
        return proposal

    def _review_knowledge_merge(
        self,
        proposal: ManagerActionProposal,
        *,
        context: ManagerActionContext,
    ) -> ManagerActionProposal:
        """Remove unsupported model-added claims from an existing-source update."""

        if proposal.action_type != "update_knowledge" or not context.maintained_knowledge:
            return proposal
        payload = {
            "maintained_knowledge": [dict(item) for item in context.maintained_knowledge],
            "manager_input": proposal.manager_input,
            "proposed_knowledge": proposal.new_value,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "<manager_merge_data>\n"
                            f"{json.dumps(payload, ensure_ascii=False)}\n"
                            "</manager_merge_data>"
                        )
                    }
                ],
            }
        ]
        turn = self.bedrock.converse(
            messages=messages,
            system_prompt=MANAGER_MERGE_REVIEW_PROMPT,
            tools=[MANAGER_MERGE_REVIEW_TOOL],
            max_tokens=1_000,
            temperature=0,
        )
        usage = dict(proposal.usage)
        self._add_usage(usage, turn)
        reviews = self._named_submissions(turn, "review_manager_merge")
        if len(reviews) != 1:
            return replace(
                proposal,
                needs_clarification=True,
                clarification_question=(
                    "The complete merged value could not be verified. Please review or restate it."
                ),
                usage=usage,
            )

        review = reviews[0]
        valid = review.get("valid")
        corrected = review.get("corrected_value")
        additions = review.get("unsupported_additions")
        if not isinstance(valid, bool) or not isinstance(corrected, str) or not isinstance(
            additions, list
        ):
            raise ManagerActionError("Manager merge review is invalid.")
        warnings = tuple(
            re.sub(r"\s+", " ", str(value)).strip()[:300]
            for value in additions
            if str(value).strip()
        )
        if valid:
            return replace(proposal, usage=usage, merge_warnings=())
        if not corrected.strip():
            return replace(
                proposal,
                needs_clarification=True,
                clarification_question=(
                    "The proposed merge contained unsupported additions and could not be repaired. "
                    "Please provide the complete intended value."
                ),
                usage=usage,
                merge_warnings=warnings,
            )
        return replace(
            proposal,
            new_value=re.sub(r"\s+", " ", corrected).strip()[:2_000],
            needs_clarification=False,
            clarification_question="",
            usage=usage,
            merge_warnings=warnings,
        )

    @classmethod
    def _complete_explicit_proposal(
        cls,
        proposal: ManagerActionProposal,
        *,
        context: ManagerActionContext,
    ) -> ManagerActionProposal:
        """Fill backend-owned context and remove contradictory clarification."""

        scope = proposal.scope.strip()
        if not scope:
            wiki_paths = [
                str(citation.get("wiki_path", "")).strip()
                for citation in context.citations
                if isinstance(citation, Mapping)
            ]
            scope = ", ".join(dict.fromkeys(path for path in wiki_paths if path))

        previous_value = proposal.previous_value.strip()
        if proposal.action_type == "update_knowledge" and not previous_value:
            previous_value = context.answer.strip()

        proposal = replace(
            proposal,
            scope=scope,
            previous_value=previous_value,
        )

        missing: list[str] = []
        if not proposal.subject:
            missing.append("subject")
        if not proposal.new_value:
            missing.append("new or corrected fact")
        if proposal.action_type == "update_knowledge" and not proposal.previous_value:
            missing.append("current value")

        if missing:
            return replace(
                proposal,
                needs_clarification=True,
                clarification_question=(
                    "Please provide the missing factual detail: " + ", ".join(missing) + "."
                ),
            )

        question = proposal.clarification_question.strip()
        backend_owned_question = (
            not question
            or cls._ACTION_CHOICE_QUESTION.search(question) is not None
            or cls._SCOPE_QUESTION.search(question) is not None
        )
        if proposal.needs_clarification and backend_owned_question:
            return replace(
                proposal,
                needs_clarification=False,
                clarification_question="",
            )
        return proposal

    @classmethod
    def _action_details(
        cls,
        message: str,
        *,
        required_action_type: str,
        draft: ManagerActionProposal | None,
    ) -> str:
        if draft is not None:
            return message.strip()
        details = message.strip()
        while match := cls._ACTION_COMMAND.match(details):
            if cls._ACTION_TYPES_BY_COMMAND[match.group("command").casefold()] != required_action_type:
                break
            details = details[match.end():].lstrip(" :")
        return details

    @staticmethod
    def _clarification_proposal(
        action_type: str,
        *,
        draft: ManagerActionProposal | None,
        usage: Mapping[str, int],
    ) -> ManagerActionProposal:
        questions = {
            "fix_answer": "What should the corrected answer say? A short sentence is enough.",
            "update_knowledge": (
                "What single new value should replace the current value, and when does it apply? "
                "A short sentence is enough."
            ),
            "add_knowledge": (
                "What new fact should be added, and who or what does it apply to? "
                "A short sentence is enough."
            ),
        }
        if draft is not None:
            return replace(
                draft,
                needs_clarification=True,
                clarification_question=questions[action_type],
                usage={str(key): int(value) for key, value in usage.items()},
            )
        return ManagerActionProposal(
            action_id=uuid.uuid4().hex[:12],
            action_type=action_type,
            subject="",
            previous_value="",
            new_value="",
            scope="",
            effective_period="",
            reason="",
            needs_clarification=True,
            clarification_question=questions[action_type],
            language="other",
            usage={str(key): int(value) for key, value in usage.items()},
        )

    @staticmethod
    def _draft_payload(draft: ManagerActionProposal | None) -> dict[str, object] | None:
        if draft is None:
            return None
        return {
            "action_id": draft.action_id,
            "action_type": draft.action_type,
            "subject": draft.subject,
            "previous_value": draft.previous_value,
            "new_value": draft.new_value,
            "scope": draft.scope,
            "effective_period": draft.effective_period,
            "reason": draft.reason,
            "manager_input": draft.manager_input,
        }

    @staticmethod
    def _submissions(turn: ConverseTurn) -> list[Mapping[str, Any]]:
        return ManagerActionInterpreter._named_submissions(turn, "submit_manager_action")

    @staticmethod
    def _named_submissions(turn: ConverseTurn, tool_name: str) -> list[Mapping[str, Any]]:
        values: list[Mapping[str, Any]] = []
        for block in turn.message.get("content", []):
            tool_use = block.get("toolUse") if isinstance(block, Mapping) else None
            if not isinstance(tool_use, Mapping) or tool_use.get("name") != tool_name:
                continue
            inputs = tool_use.get("input")
            if isinstance(inputs, Mapping):
                values.append(inputs)
        return values

    def _proposal(
        self,
        inputs: Mapping[str, Any],
        *,
        draft: ManagerActionProposal | None,
        usage: Mapping[str, int],
        required_action_type: str,
    ) -> ManagerActionProposal | None:
        is_action = inputs.get("is_manager_action")
        if not isinstance(is_action, bool):
            raise ManagerActionError("Manager action classification is invalid.")

        def field(name: str, *, limit: int = 2_000) -> str:
            value = inputs.get(name)
            candidate = str(value).strip() if isinstance(value, str) else ""
            if not candidate and draft is not None:
                candidate = str(getattr(draft, name)).strip()
            return re.sub(r"\s+", " ", candidate)[:limit].strip()

        action_type = required_action_type
        needs_clarification = inputs.get("needs_clarification")
        if not isinstance(needs_clarification, bool):
            raise ManagerActionError("Manager action clarification state is invalid.")
        language = inputs.get("language", draft.language if draft else "other")
        if language not in {"english", "italian", "other"}:
            language = draft.language if draft else "other"
        proposal = ManagerActionProposal(
            action_id=draft.action_id if draft else uuid.uuid4().hex[:12],
            action_type=str(action_type),
            subject=field("subject", limit=300),
            previous_value=field("previous_value"),
            new_value=field("new_value"),
            scope=field("scope", limit=500),
            effective_period=field("effective_period", limit=300),
            reason=field("reason"),
            needs_clarification=needs_clarification or action_type == "unclear",
            clarification_question=field("clarification_question", limit=500),
            language=str(language),
            usage={str(key): int(value) for key, value in usage.items()},
            manager_input=draft.manager_input if draft else "",
            source_path=draft.source_path if draft else None,
            feedback_path=draft.feedback_path if draft else None,
            pages_updated=draft.pages_updated if draft else (),
        )
        if not proposal.ready_for_confirmation and not proposal.needs_clarification:
            proposal = replace(proposal, needs_clarification=True)
        if proposal.needs_clarification and not proposal.clarification_question:
            proposal = replace(
                proposal,
                clarification_question=(
                    "Please provide the missing factual detail needed for this selected action."
                ),
            )
        return proposal

    @staticmethod
    def _add_usage(usage: dict[str, int], turn: ConverseTurn) -> None:
        for key, value in turn.usage.items():
            usage[key] = usage.get(key, 0) + int(value)


class ManagerActionStore:
    """Persist stable manager knowledge and answer-fix audit records."""

    WIKI_PATH_PATTERN = re.compile(
        r"(?:sources|concepts|entities|syntheses)/[^\s`'\"<>]+?\.md",
        flags=re.IGNORECASE,
    )

    def __init__(self, repository: WikiRepository) -> None:
        self.repository = repository

    def persist_knowledge(
        self,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
        *,
        approved_by: str = "POC manager",
    ) -> ManagerKnowledgeWrite:
        if not proposal.changes_knowledge:
            raise ManagerActionError("Answer fixes are not raw knowledge sources.")
        approved_at = datetime.now(timezone.utc)
        existing_sources = self._existing_manager_sources(proposal, context)
        if proposal.source_path:
            normalized = self.repository.normalize_source_path(proposal.source_path)
            if normalized.casefold().startswith(
                f"raw/{self.repository.MANAGER_KNOWLEDGE_DIR}/".casefold()
            ):
                existing_sources.add(normalized)
        if len(existing_sources) > 1:
            raise ManagerActionError(
                "The update refers to multiple manager knowledge sources; narrow its scope."
            )

        if existing_sources:
            source_path = next(iter(existing_sources))
            filename = PurePosixPath(source_path).name
        else:
            filename = f"{self._stable_key(proposal)}.md"
        content = self._knowledge_markdown(proposal, approved_by, approved_at)
        try:
            source_path, previous_content = self.repository.write_manager_knowledge_source(
                filename,
                content,
                create_only=proposal.action_type == "add_knowledge",
            )
        except RepositoryError as exc:
            raise ManagerActionError(str(exc)) from exc
        return ManagerKnowledgeWrite(source_path, previous_content)

    def rollback_knowledge(self, write: ManagerKnowledgeWrite) -> None:
        """Restore the previous stable source after failed Wiki integration."""

        self.repository.restore_manager_knowledge_source(
            write.source_path,
            write.previous_content,
        )

    def _existing_manager_sources(
        self,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
    ) -> set[str]:
        prefix = f"raw/{self.repository.MANAGER_KNOWLEDGE_DIR}/".casefold()
        sources: set[str] = set()
        page_paths: list[str] = []

        for match in self.WIKI_PATH_PATTERN.findall(proposal.scope):
            try:
                normalized = self.repository.normalize_wiki_path(match, allow_system=False)
            except RepositoryError:
                continue
            if normalized not in page_paths:
                page_paths.append(normalized)

        for citation in context.citations:
            if not isinstance(citation, Mapping):
                continue
            raw_sources = citation.get("source_paths", [])
            if isinstance(raw_sources, list):
                for value in raw_sources:
                    try:
                        source = self.repository.normalize_source_path(str(value))
                    except RepositoryError:
                        continue
                    if source.casefold().startswith(prefix) and self.repository.raw_exists(source):
                        sources.add(source)
            wiki_path = citation.get("wiki_path")
            if isinstance(wiki_path, str):
                try:
                    normalized = self.repository.normalize_wiki_path(
                        wiki_path, allow_system=False
                    )
                except RepositoryError:
                    continue
                if normalized not in page_paths:
                    page_paths.append(normalized)

        for page_path in page_paths:
            try:
                page_sources = self.repository.page_source_paths(page_path)
            except RepositoryError:
                continue
            sources.update(
                source
                for source in page_sources
                if source.casefold().startswith(prefix)
            )
        return sources

    def _stable_key(self, proposal: ManagerActionProposal) -> str:
        scope_paths = self.WIKI_PATH_PATTERN.findall(proposal.scope)
        seed = PurePosixPath(scope_paths[0]).stem if scope_paths else proposal.subject
        normalized = unicodedata.normalize("NFKD", seed).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")
        if not slug:
            slug = f"manager-knowledge-{proposal.action_id}"
        return slug[:120].rstrip("-")

    def persist_answer_fix(
        self,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
        result: Mapping[str, object],
        *,
        approved_by: str = "POC manager",
    ) -> str:
        approved_at = datetime.now(timezone.utc)
        relative = f"feedback/answer-fixes/{approved_at.strftime('%Y%m%dT%H%M%SZ')}-{proposal.action_id}.json"
        target = self.repository.backend_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "action_id": proposal.action_id,
            "action_type": "fix_answer",
            "status": "applied",
            "changes_knowledge": False,
            "approved_by": approved_by,
            "approved_at": approved_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "question": context.question,
            "wrong_answer": context.answer,
            "manager_correction": proposal.new_value,
            "subject": proposal.subject,
            "reason": proposal.reason,
            "result": dict(result),
        }
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise ManagerActionError("Answer-fix audit record already exists.") from exc
        except OSError as exc:
            raise ManagerActionError("Could not persist the answer-fix audit record.") from exc
        return relative

    @staticmethod
    def _knowledge_markdown(
        proposal: ManagerActionProposal,
        approved_by: str,
        approved_at: datetime,
    ) -> str:
        return f"""# Manager Knowledge: {proposal.subject}

- Last action ID: `{proposal.action_id}`
- Last action type: `{proposal.action_type}`
- Updated by: {approved_by}
- Updated at: {approved_at.isoformat(timespec="seconds").replace("+00:00", "Z")}
- Scope: {proposal.scope}
- Effective period: {proposal.effective_period or "Not specified"}
- Reason: {proposal.reason or "Manager-approved POC action"}

## Current approved knowledge

{proposal.new_value}
"""
