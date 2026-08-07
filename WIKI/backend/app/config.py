"""Configuration loading for AWS Bedrock.

The local JSON file is supported for development because this project already
uses it.  Environment variables take precedence, and omitting explicit access
keys leaves authentication to Boto3's normal credential provider chain (for
example an IAM role, AWS SSO, or a shared credentials profile).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class ConfigurationError(RuntimeError):
    """Raised when Bedrock configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class BedrockSettings:
    """Non-secret and secret configuration required by the backend.

    Secret fields are excluded from the generated repr so an exception or
    debug log cannot accidentally print credentials.
    """

    project_root: Path
    region_name: str | None
    bedrock_model_id: str
    aws_access_key_id: str | None = field(default=None, repr=False)
    aws_secret_access_key: str | None = field(default=None, repr=False)
    aws_session_token: str | None = field(default=None, repr=False)
    max_source_bytes: int = 25_000_000
    max_extracted_characters: int = 600_000
    max_agent_steps: int = 24
    max_output_tokens: int = 4096

    @property
    def backend_root(self) -> Path:
        return self.project_root / "backend"

    @property
    def credentials_source(self) -> str:
        return "explicit" if self.aws_access_key_id else "default-chain"

    @property
    def is_configured(self) -> bool:
        # Region may also come from the Boto3 shared config/default chain.
        return bool(self.bedrock_model_id)


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _read_credentials_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        # Do not include file contents (which contain secrets) in the error.
        raise ConfigurationError(f"Could not read valid JSON configuration from {path.name}.") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{path.name} must contain a JSON object.")
    return payload


def _first(*values: object) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            return cleaned
    return None


def load_settings(
    project_root: Path | str | None = None,
    *,
    credentials_path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> BedrockSettings:
    """Load Bedrock settings without constructing an AWS client.

    Precedence is environment, ``aws_credentials.json``, then (for AWS
    credentials) Boto3's default chain.  The model ID is application-specific
    and therefore must be supplied either through ``BEDROCK_MODEL_ID`` or the
    JSON file.
    """

    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    config_path = Path(credentials_path).resolve() if credentials_path else root / "aws_credentials.json"
    env = os.environ if environ is None else environ
    file_values = _read_credentials_file(config_path)

    region_name = _first(
        env.get("AWS_REGION"),
        env.get("AWS_DEFAULT_REGION"),
        file_values.get("region_name"),
    )
    model_id = _first(env.get("BEDROCK_MODEL_ID"), file_values.get("bedrock_model_id"))
    env_access_key = _clean(env.get("AWS_ACCESS_KEY_ID"))
    env_secret_key = _clean(env.get("AWS_SECRET_ACCESS_KEY"))
    env_session_token = _clean(env.get("AWS_SESSION_TOKEN"))
    if env_access_key or env_secret_key or env_session_token:
        # Treat an environment credential set atomically. Mixing one key from
        # the environment with another from JSON is both surprising and unsafe.
        access_key = env_access_key
        secret_key = env_secret_key
        session_token = env_session_token
    else:
        access_key = _clean(file_values.get("aws_access_key_id"))
        secret_key = _clean(file_values.get("aws_secret_access_key"))
        session_token = _clean(file_values.get("aws_session_token"))

    if not model_id:
        raise ConfigurationError(
            "A Bedrock model is required. Set BEDROCK_MODEL_ID or bedrock_model_id in aws_credentials.json."
        )
    if bool(access_key) != bool(secret_key):
        raise ConfigurationError(
            "Explicit AWS credentials are incomplete; provide both access key ID and secret access key."
        )
    if session_token and not access_key:
        raise ConfigurationError("AWS_SESSION_TOKEN requires explicit access key credentials.")

    return BedrockSettings(
        project_root=root,
        region_name=region_name,
        bedrock_model_id=model_id,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )
