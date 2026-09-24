"""Upstream layer: fingerprint headers, body variants, HTTP client, error model.

All shapes here come from the Phase-1 capture. Do not "improve" the header set
or the body variants without re-capturing — the upstream fingerprints the client.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

import httpx

from .config import UpstreamConfig

log = logging.getLogger("cline_gateway.upstream")

# --------------------------------------------------------------------------- #
# headers
# --------------------------------------------------------------------------- #

# The 16 static headers observed on every captured call. `authorization` and
# `x-task-id` are filled per request; `host` and `content-length` are set by httpx.
STATIC_HEADERS: dict[str, str] = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "connection": "keep-alive",
    "content-type": "application/json",
    "http-referer": "https://cline.bot",
    "x-client-type": "cline-desktop",
    "x-is-multiroot": "false",
    "x-title": "Cline",
}

# header names that must be present on every outbound request (regression guard)
REQUIRED_HEADER_NAMES = {
    "accept", "accept-encoding", "authorization", "connection", "content-type",
    "http-referer", "user-agent", "x-client-type", "x-client-version",
    "x-core-version", "x-is-multiroot", "x-platform", "x-platform-version",
    "x-task-id", "x-title",
}


def new_task_id() -> str:
    return f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:5]}"


def build_headers(access_token: str, fp, task_id: str | None = None) -> dict[str, str]:
    h = dict(STATIC_HEADERS)
    h["user-agent"] = fp.user_agent
    h["x-client-version"] = fp.client_version
    h["x-core-version"] = fp.core_version
    h["x-platform"] = fp.platform
    h["x-platform-version"] = fp.client_version
    h["x-task-id"] = task_id or new_task_id()
    h["authorization"] = access_token if access_token.lower().startswith("bearer ") \
        else f"Bearer {access_token}"
    return h


# --------------------------------------------------------------------------- #
# body variants
# --------------------------------------------------------------------------- #

VARIANTS = ("default", "anthropic", "openai-nextgen", "reasoning")

# Effort values clients send, normalised to what the upstream accepts.
# `none`/`off` disable thinking; `extra` is the Cline picker's top level and
# the wire value is `xhigh` (captured 2026-09-24). `minimal` sits below every
# observed family floor. `low` is real for gemini-3.8-flash (captured) but was
# never observed for kimi-k3, whose client floor is `medium` — that one stays
# family-specific in _normalise_effort.
_EFFORT_ALIASES = {
    "none": "none", "off": "none", "disabled": "none", "false": "none",
    "minimal": "medium",
    "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh",
    "extra": "xhigh", "max": "xhigh",
}


def _normalise_effort(requested: str, model: str = "") -> str | None:
    """Client effort string -> the upstream value it maps to, or None."""
    mapped = _EFFORT_ALIASES.get(requested)
    if requested == "low" and "kimi" in (model or "").lower():
        return "medium"          # kimi family floor (never observed below)
    return mapped


def _client_effort(payload: dict[str, Any]) -> str | None:
    """Reasoning control a client asked for, whatever spelling it used.

    Accepts the OpenAI snake_case key, the camelCase form Kilo Code's variant
    bodies use, an output-effort key, and the enable_thinking / thinking
    shapes its settings UI can emit.
    """
    for key in ("reasoning_effort", "reasoningEffort", "effort"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        ttype = str(thinking.get("type") or "").lower()
        if ttype in ("disabled", "off"):
            return "none"
        if ttype in ("enabled", "adaptive", "auto", "on"):
            return "medium"

    enable = payload.get("enable_thinking")
    if enable is False:
        return "none"
    if enable is True:
        return "medium"
    return None


def build_upstream_body(payload: dict[str, Any], variant: str,
                        default_max_tokens: int = 32000,
                        anthropic_cache_control: bool = True) -> dict[str, Any]:
    """Turn an internal (OpenAI-shaped) payload into the captured upstream body."""
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = default_max_tokens

    body: dict[str, Any] = {
        "model": payload["model"],
        "messages": payload.get("messages", []),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice", "auto"),
        # every captured call used stream=true; the gateway always streams
        # upstream and aggregates locally for non-streaming clients.
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if variant == "reasoning":
        # Captured 2026-09-19 (cline-free/kimi-k3) and 2026-09-24
        # (cline-free/gemini-3.8-flash): this family sends NEITHER max_tokens
        # NOR max_completion_tokens, and drives thinking with either
        # {"reasoning": {"enabled": false}} or a reasoning_effort of
        # "low" | "medium" | "high" | "xhigh".
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict) and not reasoning.get("effort"):
            body["reasoning"] = reasoning
        else:
            requested = _client_effort(payload)
            if requested is None:
                # an explicit reasoning object with an effort we cannot map,
                # or nothing at all: the captured default is thinking off
                if isinstance(reasoning, dict):
                    body["reasoning"] = reasoning
                else:
                    body["reasoning"] = {"enabled": False}
            else:
                mapped = _normalise_effort(requested, body["model"])
                if mapped is None:
                    # unknown effort string: coerce to the family's floor and
                    # say so — silent rewriting made client intent invisible
                    log.warning("unknown reasoning_effort %r; using 'medium'",
                                requested)
                    mapped = "medium"
                if mapped == "none":
                    body["reasoning"] = {"enabled": False}
                else:
                    body["reasoning_effort"] = mapped
    else:
        # Captured 2026-09-24 (space-bunny-alpha, mimo-v2.6-flash,
        # deepseek-v4.1-flash, muse-spark): token limit + reasoning_effort side
        # by side; the "None" picker level is {"reasoning": {"enabled": false}}.
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict) and not reasoning.get("effort"):
            body["reasoning"] = reasoning
        else:
            requested = _client_effort(payload)
            if requested is None:
                if isinstance(reasoning, dict):
                    body["reasoning"] = reasoning
                else:
                    body["reasoning_effort"] = "low"     # captured default
            else:
                mapped = _normalise_effort(requested, body["model"])
                if mapped is None:
                    log.warning("unknown reasoning_effort %r; using 'low'",
                                requested)
                    mapped = "low"
                if mapped == "none":
                    body["reasoning"] = {"enabled": False}
                else:
                    body["reasoning_effort"] = mapped
        if variant == "openai-nextgen":
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens

    if variant == "anthropic" and anthropic_cache_control:
        body["cache_control"] = {"type": "ephemeral"}

    # Forward the standard sampling controls clients actually set. The captured
    # Cline client never sent them, but dropping them made temperature/top_p/
    # stop silently no-ops for every other OpenAI/Anthropic client (the
    # Anthropic translator produced them; this body then discarded them).
    # top_k is Anthropic-specific but included: silently ignoring a sampling
    # parameter is worse than a client-visible 400 if the upstream rejects it.
    for key in ("temperature", "top_p", "top_k", "stop", "presence_penalty",
                "frequency_penalty", "seed", "response_format", "logit_bias"):
        if payload.get(key) is not None:
            body[key] = payload[key]

    # `tools` is omitted entirely when absent in the captured calls
    if body["tools"] is None:
        del body["tools"]

    return body


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class ErrorKind(str, Enum):
    OK = "ok"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    UNAUTHORIZED = "unauthorized"
    ENTITLEMENT = "entitlement"     # 403: no plan/subscription for THIS model
    EDGE_BLOCK = "edge_block"       # 403 HTML from the CDN/edge, not the API
    RATE_LIMITED = "rate_limited"
    SERVER = "server"
    CLIENT = "client"
    TRANSPORT = "transport"


# Codes that mean "this account cannot use this model", not "this account is bad".
ENTITLEMENT_CODES = frozenset({
    "ENTITLEMENT_ERROR",
    "NOT_SUBSCRIBED",
    "PLAN_REQUIRED",
    "MODEL_NOT_IN_PLAN",
})


def is_entitlement_code(code: str) -> bool:
    """Single source of truth for entitlement classification.

    Three call sites used to classify independently (this frozenset plus two
    substring checks), and they had already drifted: NOT_SUBSCRIBED matched
    here but not the substring test. Accepts exact codes and any code
    containing ENTITLEMENT/SUBSCRIB/PLAN (covers future plan-gate codes).
    """
    c = (code or "").upper()
    if not c:
        return False
    if c in ENTITLEMENT_CODES:
        return True
    return ("ENTITLEMENT" in c) or ("SUBSCRIB" in c) or ("PLAN" in c)


def _error_code(body: str | bytes | None) -> str:
    """Best-effort extraction of error.code from a JSON error body."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
    try:
        parsed = json.loads(text)
    except Exception:
        return ""
    err = parsed.get("error", parsed)
    if isinstance(err, dict):
        return str(err.get("code") or "").upper()
    return ""


# Captured 2026-09-16: a global 403 served as HTML by Google's front end, for every
# model regardless of entitlement:
#   <html>...<title>403 Forbidden</title>...
#   <h2>Your client does not have permission to get URL
#       <code>/api/v1/chat/completions</code> from this server.</h2>
# This is infrastructure, not the API, and not the account's fault.
EDGE_BLOCK_MARKERS = (
    "does not have permission to get url",
    "403 forbidden",
    "error: forbidden",
)


def _looks_like_edge_block(body: str | bytes | None) -> bool:
    text = (body.decode("utf-8", "replace") if isinstance(body, bytes)
            else (body or "")).lower()
    if "<html" not in text:
        return False
    return any(marker in text for marker in EDGE_BLOCK_MARKERS)


def classify(status: int, body: str | bytes | None = None) -> ErrorKind:
    if status == 200:
        return ErrorKind.OK
    if status == 401:
        return ErrorKind.UNAUTHORIZED
    if status == 403:
        if is_entitlement_code(_error_code(body)):
            return ErrorKind.ENTITLEMENT
        if _looks_like_edge_block(body):
            return ErrorKind.EDGE_BLOCK
        return ErrorKind.CLIENT
    if status == 402:
        return ErrorKind.INSUFFICIENT_CREDITS
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if 500 <= status < 600:
        return ErrorKind.SERVER
    return ErrorKind.CLIENT


def edge_block_message(body: str | bytes | None) -> str:
    """Short, human-readable summary of an edge block (bodies are HTML)."""
    text = (body.decode("utf-8", "replace") if isinstance(body, bytes)
            else (body or ""))
    lowered = text.lower()
    if "does not have permission to get url" in lowered:
        return ("upstream edge rejected the request (403): the client has no "
                "permission for this URL")
    return "upstream edge rejected the request (403 Forbidden)"


def parse_upstream_error(status: int, body: str | bytes | None) -> dict[str, Any]:
    """Normalise an upstream error body (captured shape) into a dict."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
    try:
        parsed = json.loads(text)
        err = parsed.get("error", parsed)
        if isinstance(err, dict):
            return {
                "code": err.get("code") or f"http_{status}",
                "message": err.get("message") or text[:400],
                "raw": parsed,
            }
    except Exception:
        pass
    return {"code": f"http_{status}", "message": text[:400], "raw": None}


# Mapping for the OpenAI-compatible error envelope.
ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "insufficient_quota",
    403: "permission_error",
    404: "not_found_error",
    413: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    502: "server_error",
    503: "server_error",
    504: "server_error",
}


def openai_error(status: int, code: str, message: str,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    error = {
        "message": message,
        "type": ERROR_TYPE_BY_STATUS.get(status, "api_error"),
        "param": None,
        "code": code,
    }
    if extra:
        error.update(extra)
    return {"error": error}


def anthropic_error(status: int, code: str, message: str,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    # Anthropic error types: invalid_request_error, authentication_error,
    # permission_error, not_found_error, request_too_large, rate_limit_error,
    # api_error, overloaded_error
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
    }
    if status >= 500:
        etype = "overloaded_error" if status == 529 else "api_error"
    else:
        etype = mapping.get(status, "api_error")
    error: dict[str, Any] = {"type": etype, "message": message}
    # Anthropic's envelope has no standard code field, but clients doing
    # programmatic retry need the machine-readable code the OpenAI dialect
    # already exposes — carry it as an additive field (ignored by strict clients)
    if code:
        error["code"] = code
    if extra:
        error.update(extra)
    return {"type": "error", "error": error}


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #


@dataclass
class UpstreamResponse:
    status_code: int
    headers: dict[str, str]
    _resp: httpx.Response

    async def aread(self) -> bytes:
        return await self._resp.aread()

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return self._resp.aiter_bytes()

    async def aclose(self) -> None:
        await self._resp.aclose()


class UpstreamClient:
    """Thin httpx wrapper. Adds nothing to the request the caller did not set."""

    def __init__(self, cfg: UpstreamConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.timeout_read, connect=cfg.timeout_connect),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            headers={},          # never inject a default User-Agent
            follow_redirects=False,
            http2=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send(self, headers: dict[str, str], body: dict[str, Any],
                   stream: bool) -> UpstreamResponse:
        req = self._client.build_request(
            "POST", self.cfg.chat_url, headers=headers, json=body,
        )
        try:
            resp = await self._client.send(req, stream=stream)
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc
        return UpstreamResponse(
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items()},
            _resp=resp,
        )


class TransportError(Exception):
    """Network-level failure — safe to retry on another account."""


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #


def iter_sse_events(chunk_text: str) -> list[dict[str, Any] | str]:
    """Parse an SSE buffer into events.

    Returns dicts for `data: {...}` JSON frames and the string "[DONE]" for the
    terminator. Non-data lines (comments, blank lines) are ignored.
    """
    events: list[dict[str, Any] | str] = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return events


def sse_frame(obj: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
            + "\n\n").encode("utf-8")
