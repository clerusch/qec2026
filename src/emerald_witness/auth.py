from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from .config import DEFAULT_TOKEN_ENV_VARS


def normalize_token(token: str) -> str:
    normalized = token.strip().strip("'\"")
    if normalized.lower().startswith("bearer "):
        normalized = normalized[7:].strip()
    if not normalized:
        raise RuntimeError("Resolved IQM token was empty after normalization.")
    return normalized


def resolve_token(
    token: str | None,
    env_var_names: tuple[str, ...] = DEFAULT_TOKEN_ENV_VARS,
) -> str:
    if token:
        return normalize_token(token)

    for env_var in env_var_names:
        value = os.getenv(env_var)
        if value:
            return normalize_token(value)

    env_list = ", ".join(env_var_names)
    raise RuntimeError(f"No IQM API token found. Set one of {env_list}, or pass --token.")


@contextmanager
def explicit_token_environment(token: str) -> Iterator[str]:
    previous = {env_var: os.environ.pop(env_var, None) for env_var in DEFAULT_TOKEN_ENV_VARS}
    try:
        yield token
    finally:
        for env_var, value in previous.items():
            if value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = value
