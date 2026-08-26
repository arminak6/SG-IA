"""Private per-user preferences and session-scoped conversation persistence."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MAX_PREFERENCES = 20
MAX_PREFERENCE_CHARACTERS = 500
MAX_CONTEXT_MESSAGES = 24
MAX_CONTEXT_CHARACTERS = 16_000

FORGET_PREFERENCES_PATTERN = re.compile(
    r"^(?:/forget(?:\s+(?:all\s+)?)?preferences?|forget\s+(?:all\s+)?my\s+preferences|"
    r"dimentica\s+(?:tutte\s+)?le\s+mie\s+preferenze)\s*[.!]?$",
    flags=re.IGNORECASE,
)
EXPLICIT_REMEMBER_PATTERN = re.compile(
    r"^/remember\s+(.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)
DURABLE_PREFERENCE_PATTERN = re.compile(
    r"^(?:please\s+)?always\b|^(?:from\s+now\s+on|remember(?:\s+that)?|"
    r"i\s+prefer|my\s+preference\s+is)\b|"
    r"^(?:per\s+favore\s+)?(?:rispondimi|rispondi)\s+sempre\b|"
    r"^(?:d['']ora\s+in\s+poi|ricorda(?:ti)?(?:\s+che)?|preferisco)\b",
    flags=re.IGNORECASE,
)


class UserMemoryError(ValueError):
    """Raised when user-memory input or persisted state is invalid."""


@dataclass(frozen=True)
class PreferenceInstruction:
    action: str
    value: str | None = None


def detect_preference_instruction(message: str) -> PreferenceInstruction | None:
    """Recognize explicit durable preference requests without classifying normal chat."""

    cleaned = " ".join(str(message).strip().split())
    if not cleaned:
        return None
    if FORGET_PREFERENCES_PATTERN.fullmatch(cleaned):
        return PreferenceInstruction(action="clear")
    explicit = EXPLICIT_REMEMBER_PATTERN.fullmatch(cleaned)
    if explicit:
        value = explicit.group(1).strip()
        return PreferenceInstruction(action="add", value=value) if value else None
    if DURABLE_PREFERENCE_PATTERN.search(cleaned):
        return PreferenceInstruction(action="add", value=cleaned)
    return None


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    preferences: tuple[str, ...]
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "preferences": list(self.preferences),
            "updated_at": self.updated_at,
        }


class UserMemoryStore:
    """Store private user data outside the Wiki knowledge and search roots."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    @staticmethod
    def normalize_user_id(user_id: str) -> str:
        normalized = str(user_id).strip()
        if not USER_ID_PATTERN.fullmatch(normalized):
            raise UserMemoryError(
                "user_id must contain only letters, numbers, '.', '_' or '-' "
                "and be at most 64 characters"
            )
        return normalized

    @staticmethod
    def normalize_session_id(session_id: str) -> str:
        normalized = str(session_id).strip()
        if not SESSION_ID_PATTERN.fullmatch(normalized):
            raise UserMemoryError(
                "session_id must contain only letters, numbers, '.', '_' or '-' "
                "and be at most 128 characters"
            )
        return normalized

    @staticmethod
    def normalize_preferences(preferences: Sequence[str]) -> tuple[str, ...]:
        if isinstance(preferences, (str, bytes)):
            raise UserMemoryError("preferences must be a list of strings")
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_value in preferences:
            if not isinstance(raw_value, str):
                raise UserMemoryError("each preference must be a string")
            value = " ".join(raw_value.strip().split())
            if not value:
                continue
            if len(value) > MAX_PREFERENCE_CHARACTERS:
                raise UserMemoryError(
                    f"each preference must be at most {MAX_PREFERENCE_CHARACTERS} characters"
                )
            folded = value.casefold()
            if folded not in seen:
                cleaned.append(value)
                seen.add(folded)
        if len(cleaned) > MAX_PREFERENCES:
            raise UserMemoryError(f"at most {MAX_PREFERENCES} preferences are allowed")
        return tuple(cleaned)

    def get_profile(self, user_id: str) -> UserProfile:
        normalized_user = self.normalize_user_id(user_id)
        path = self._profile_json_path(normalized_user)
        with self._lock:
            if not path.is_file():
                return UserProfile(user_id=normalized_user, preferences=())
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise UserMemoryError("the stored user profile is invalid") from exc
        if not isinstance(payload, Mapping):
            raise UserMemoryError("the stored user profile is invalid")
        preferences = payload.get("preferences", [])
        if not isinstance(preferences, list):
            raise UserMemoryError("the stored user preferences are invalid")
        return UserProfile(
            user_id=normalized_user,
            preferences=self.normalize_preferences(preferences),
            updated_at=(
                str(payload["updated_at"])
                if isinstance(payload.get("updated_at"), str)
                else None
            ),
        )

    def save_profile(self, user_id: str, preferences: Sequence[str]) -> UserProfile:
        normalized_user = self.normalize_user_id(user_id)
        normalized_preferences = self.normalize_preferences(preferences)
        updated_at = self._timestamp()
        profile = UserProfile(
            user_id=normalized_user,
            preferences=normalized_preferences,
            updated_at=updated_at,
        )
        user_dir = self._user_dir(normalized_user)
        with self._lock:
            user_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(self._profile_json_path(normalized_user), {
                "version": 1,
                **profile.to_dict(),
            })
            self._atomic_write_text(
                user_dir / "profile.md",
                self._render_profile(profile),
            )
        return profile

    def add_preference(self, user_id: str, preference: str) -> UserProfile:
        profile = self.get_profile(user_id)
        return self.save_profile(user_id, (*profile.preferences, preference))

    def append_exchange(
        self,
        user_id: str,
        session_id: str,
        question: str,
        response: Mapping[str, object],
        *,
        include_in_context: bool = True,
    ) -> None:
        normalized_user = self.normalize_user_id(user_id)
        normalized_session = self.normalize_session_id(session_id)
        session_path = self._session_path(normalized_user, normalized_session)
        timestamp = self._timestamp()
        user_event = {
            "timestamp": timestamp,
            "role": "user",
            "content": str(question),
            "include_in_context": include_in_context,
        }
        assistant_event = {
            "timestamp": self._timestamp(),
            "role": "assistant",
            "content": str(response.get("answer", "")),
            "status": str(response.get("status", "answered")),
            "citations": self._safe_citations(response.get("citations", [])),
            "include_in_context": include_in_context,
        }
        with self._lock:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with session_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for event in (user_event, assistant_event):
                        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise UserMemoryError("could not save the chat history") from exc

    def read_session_context(
        self,
        user_id: str,
        session_id: str,
        *,
        max_messages: int = MAX_CONTEXT_MESSAGES,
        max_characters: int = MAX_CONTEXT_CHARACTERS,
    ) -> tuple[dict[str, str], ...]:
        normalized_user = self.normalize_user_id(user_id)
        normalized_session = self.normalize_session_id(session_id)
        path = self._session_path(normalized_user, normalized_session)
        with self._lock:
            if not path.is_file():
                return ()
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise UserMemoryError("could not read the chat history") from exc

        messages: list[dict[str, str]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("include_in_context") is False:
                continue
            role = event.get("role")
            content = event.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                messages.append({"role": str(role), "content": content})

        selected: list[dict[str, str]] = []
        used_characters = 0
        for message in reversed(messages[-max_messages:]):
            length = len(message["content"])
            if selected and used_characters + length > max_characters:
                break
            if length > max_characters:
                message = dict(message)
                message["content"] = message["content"][-max_characters:]
                length = len(message["content"])
            selected.append(message)
            used_characters += length
        selected.reverse()
        return tuple(selected)

    def clear_session(self, user_id: str, session_id: str) -> bool:
        normalized_user = self.normalize_user_id(user_id)
        normalized_session = self.normalize_session_id(session_id)
        path = self._session_path(normalized_user, normalized_session)
        with self._lock:
            if not path.is_file():
                return False
            try:
                path.unlink()
            except OSError as exc:
                raise UserMemoryError("could not delete the chat history") from exc
        return True

    def _user_dir(self, user_id: str) -> Path:
        return self.root / user_id

    def _profile_json_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "preferences.json"

    def _session_path(self, user_id: str, session_id: str) -> Path:
        return self._user_dir(user_id) / "sessions" / f"{session_id}.jsonl"

    @staticmethod
    def _safe_citations(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        citations: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            citations.append(
                {
                    "wiki_path": str(item.get("wiki_path", "")),
                    "source_paths": [
                        str(path)
                        for path in item.get("source_paths", [])
                        if isinstance(path, str)
                    ]
                    if isinstance(item.get("source_paths", []), list)
                    else [],
                }
            )
        return citations

    @staticmethod
    def _render_profile(profile: UserProfile) -> str:
        lines = [
            f"# User Profile: {profile.user_id}",
            "",
            "This private profile controls response personalization only. It is not Wiki knowledge.",
            "",
            "## Preferences",
            "",
        ]
        if profile.preferences:
            lines.extend(f"- {preference}" for preference in profile.preferences)
        else:
            lines.append("- No saved preferences.")
        lines.extend(["", f"Updated: {profile.updated_at}", ""])
        return "\n".join(lines)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        UserMemoryStore._atomic_write_text(path, text)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise UserMemoryError("could not save the user profile") from exc
