"""Approved manager actions for the WIKI proof of concept."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from .bedrock import BedrockConverseClient, ConverseTurn
from .repository import WikiRepository


ACTION_TYPES = frozenset({"fix_answer", "update_knowledge", "add_knowledge"})


class ManagerActionError(RuntimeError):
    """Raised when a manager action cannot be interpreted or persisted."""


@dataclass(frozen=True)
class ManagerActionContext:
    question: str
    answer: str
    citations: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [dict(item) for item in self.citations],
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
    source_path: str | None = None
    feedback_path: str | None = None
    pages_updated: tuple[str, ...] = ()

    @property
    def changes_knowledge(self) -> bool:
        return self.action_type in {"update_knowledge", "add_knowledge"}

    @property
    def wiki_maintenance(self) -> bool:
        return self.action_type == "fix_answer"

    @property
    def ready_for_confirmation(self) -> bool:
        if self.needs_clarification or self.action_type not in ACTION_TYPES:
            return False
        if not self.subject or not self.new_value:
            return False
        if self.changes_knowledge and not self.scope:
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
            "subject": self.subject,
            "previous_value": self.previous_value,
            "corrected_value": self.new_value,
            "scope": self.scope,
            "effective_period": self.effective_period,
            "source_path": self.source_path,
            "feedback_path": self.feedback_path,
            "pages_updated": list(self.pages_updated),
        }


@dataclass(frozen=True)
class ManagerActionSession:
    context: ManagerActionContext
    pending: ManagerActionProposal | None = None


MANAGER_ACTION_TOOL = {
    "toolSpec": {
        "name": "submit_manager_action",
        "description": "Classify and structure a trusted manager action for the Wiki POC.",
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


MANAGER_ACTION_PROMPT = """You classify and structure actions from a trusted manager in a Wiki proof of concept.

Use only the application-supplied previous interaction, draft, and manager message. Choose exactly one:
- fix_answer: the Wiki already contains the correct information, but the previous LLM answer retrieved,
  interpreted, ordered, or expressed it incorrectly. This adds no source knowledge.
- update_knowledge: maintained knowledge is outdated or wrong and the manager supplies a replacement.
- add_knowledge: the manager supplies genuinely new information absent from maintained knowledge.
- unclear: the message does not distinguish these cases.

Call submit_manager_action exactly once. Never guess between fix_answer and a knowledge change: request
clarification when ambiguous. Preserve useful draft fields. For time-dependent knowledge changes, require
the effective period. A normal follow-up question is not a manager action. Never invent domain details.
"""


class ManagerActionInterpreter:
    """Interpret manager messages into one of three explicit action types."""

    _ACTION_CUES = re.compile(
        r"^\s*(?:no\b|incorrect\b|wrong\b|actually\b|correction\b|correct\s+yourself\b|"
        r"fix\s+(?:the\s+)?answer\b|update\s+(?:the\s+)?(?:wiki|knowledge)\b|"
        r"add\s+(?:new\s+)?(?:wiki\s+)?knowledge\b|new\s+knowledge\b|"
        r"that(?:'s|\s+is)\s+wrong\b|è\s+sbagliat[oa]\b|e\s+sbagliat[oa]\b|"
        r"correggi\b|correzione\b|aggiorna\b|aggiungi\b|in\s+realtà\b)",
        flags=re.IGNORECASE,
    )
    _CONFIRMATIONS = frozenset(
        {"confirm", "confirmed", "yes", "yes confirm", "approve", "apply", "confermo", "approva", "applica", "si", "sì"}
    )
    _CANCELLATIONS = frozenset(
        {"cancel", "cancel action", "cancel correction", "reject", "annulla", "rifiuta"}
    )

    def __init__(self, bedrock: BedrockConverseClient) -> None:
        self.bedrock = bedrock

    @classmethod
    def looks_like_action(cls, message: str) -> bool:
        return cls._ACTION_CUES.search(message) is not None

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

    def interpret(
        self,
        context: ManagerActionContext,
        manager_message: str,
        *,
        draft: ManagerActionProposal | None = None,
    ) -> ManagerActionProposal | None:
        payload = {
            "previous_interaction": context.to_dict(),
            "existing_draft": self._draft_payload(draft),
            "manager_message": manager_message,
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
                return self._proposal(submissions[0], draft=draft, usage=usage)
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
        raise ManagerActionError("Manager action interpreter returned no structured proposal.")

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
        }

    @staticmethod
    def _submissions(turn: ConverseTurn) -> list[Mapping[str, Any]]:
        values: list[Mapping[str, Any]] = []
        for block in turn.message.get("content", []):
            tool_use = block.get("toolUse") if isinstance(block, Mapping) else None
            if not isinstance(tool_use, Mapping) or tool_use.get("name") != "submit_manager_action":
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
    ) -> ManagerActionProposal | None:
        is_action = inputs.get("is_manager_action")
        if not isinstance(is_action, bool):
            raise ManagerActionError("Manager action classification is invalid.")
        if not is_action:
            return None

        def field(name: str, *, limit: int = 2_000) -> str:
            value = inputs.get(name)
            candidate = str(value).strip() if isinstance(value, str) else ""
            if not candidate and draft is not None:
                candidate = str(getattr(draft, name)).strip()
            return re.sub(r"\s+", " ", candidate)[:limit].strip()

        action_type = inputs.get("action_type")
        if action_type not in ACTION_TYPES | {"unclear"}:
            action_type = draft.action_type if draft else "unclear"
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
                    "Please specify whether this fixes the previous answer, updates existing "
                    "knowledge, or adds new knowledge, and provide the missing details."
                ),
            )
        return proposal

    @staticmethod
    def _add_usage(usage: dict[str, int], turn: ConverseTurn) -> None:
        for key, value in turn.usage.items():
            usage[key] = usage.get(key, 0) + int(value)


class ManagerActionStore:
    """Persist approved knowledge actions and answer-fix audit records."""

    def __init__(self, repository: WikiRepository) -> None:
        self.repository = repository

    def persist_knowledge(
        self,
        proposal: ManagerActionProposal,
        context: ManagerActionContext,
        *,
        approved_by: str = "POC manager",
    ) -> str:
        if not proposal.changes_knowledge:
            raise ManagerActionError("Answer fixes are not raw knowledge sources.")
        if proposal.source_path:
            return proposal.source_path
        approved_at = datetime.now(timezone.utc)
        filename = f"{approved_at.strftime('%Y%m%dT%H%M%SZ')}-{proposal.action_id}.md"
        content = self._knowledge_markdown(proposal, approved_by, approved_at)
        return self.repository.create_manager_action_source(filename, content)

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
        history = ""
        if proposal.action_type == "update_knowledge":
            history = f"""## Superseded value

{proposal.previous_value}

For the stated scope and effective period, the approved value below supersedes conflicting older
statements. Preserve older provenance and label the conflict as superseded.

"""
        return f"""# Approved Manager Action: {proposal.subject}

- Action ID: `{proposal.action_id}`
- Action type: `{proposal.action_type}`
- Approved by: {approved_by}
- Approved at: {approved_at.isoformat(timespec="seconds").replace("+00:00", "Z")}
- Scope: {proposal.scope}
- Effective period: {proposal.effective_period or "Not specified"}
- Reason: {proposal.reason or "Manager-approved POC action"}

{history}## Approved knowledge

{proposal.new_value}
"""
