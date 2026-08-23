"""Bearer-token protection for the endpoints that start or expose runs.

``/runs``, ``/runs/{id}``, ``/webhook/manual`` and ``/search`` were open. Anyone
who could reach the server could start agent runs on the operator's LLM credits
and GitHub token, or read every issue and diff the agent had touched.

Fails closed when ``DEVAGENT_API_TOKEN`` is unset, matching how
``_verify_signature`` already treats a missing webhook secret. An unset secret
must not silently mean "allow everyone" on endpoints with this blast radius.

``/webhook`` deliberately does *not* use this -- GitHub cannot send a bearer
token, so it stays on HMAC signature verification.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


TOKEN_ENV_VAR = "DEVAGENT_API_TOKEN"


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing ``Authorization: Bearer <token>``."""
    expected = os.getenv(TOKEN_ENV_VAR)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"{TOKEN_ENV_VAR} is not configured; refusing unauthenticated requests.",
        )

    token = _bearer_token(authorization)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header; expected 'Bearer <token>'.",
        )

    # Constant-time so a wrong token cannot be recovered by timing the response.
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API token")


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()
