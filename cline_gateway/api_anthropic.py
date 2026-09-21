"""Anthropic-compatible surface: /v1/messages (+ count_tokens).

Accepts the Anthropic Messages API, translates to the internal OpenAI-shaped
payload, forwards upstream, and renders the reply back in Anthropic format —
including full SSE event translation (message_start / content_block_* / message_stop).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .deps import client_key, get_state
from .service import NoAccountsAvailable, UpstreamFailure, aggregate_openai_stream
from .translate_anthropic import (extract_stream_error, request_to_openai,
                                 response_to_anthropic, stream_had_content)
from .upstream import anthropic_error

router = APIRouter(tags=["anthropic"])

SSE_HEADERS = {
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}


def _error(status: int, message: str, code: str = "",
           extra: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content=anthropic_error(status, code, message, extra))


@router.post("/v1/messages")
async def messages(
    request: Request,
    body: dict[str, Any] = Body(...),
    key: str = Depends(client_key),
):
    state = get_state(request)

    messages = body.get("messages")
    if not (isinstance(messages, list) and messages
            and all(isinstance(m, dict) and isinstance(m.get("role"), str)
                    for m in messages)):
        return _error(400, "'messages' must be a non-empty list of message "
                           "objects, each with a string 'role'")

    upstream_model = state.registry.resolve(body.get("model"), "anthropic")
    variant = state.registry.variant_for(upstream_model)
    try:
        payload = request_to_openai(body, upstream_model,
                                    state.cfg.models.default_max_tokens)
    except ValueError as exc:
        return _error(400, str(exc))

    wants_stream = bool(body.get("stream", False))

    try:
        if wants_stream:
            # see api_openai: open upstream first so a failure can still be
            # reported with a real status instead of a dropped connection
            handle = await state.service.open_stream(
                payload, model=upstream_model, variant=variant,
                dialect="anthropic", client_key=key,
            )
            generator = state.service.stream_from(
                handle, model=upstream_model, dialect="anthropic",
                client_key=key, anthropic=True,
            )
            return StreamingResponse(generator, media_type="text/event-stream",
                                     headers=SSE_HEADERS)

        result = await state.service.complete(
            payload, model=upstream_model, variant=variant,
            dialect="anthropic", client_key=key,
        )
        text = result.raw.decode("utf-8", "replace")

        stream_err = extract_stream_error(text)
        if stream_err and not stream_had_content(text):
            return _error(502, stream_err["message"],
                          str(stream_err.get("code") or ""))

        chat = aggregate_openai_stream(text, upstream_model,
                                       f"chatcmpl-{uuid.uuid4().hex[:24]}")
        return JSONResponse(response_to_anthropic(chat, upstream_model))

    except NoAccountsAvailable:
        return _error(503, "no usable Cline account in the pool",
                      "NO_USABLE_ACCOUNTS")
    except UpstreamFailure as exc:
        return _error(exc.status, exc.message, exc.code, exc.extra())


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    body: dict[str, Any] = Body(...),
    key: str = Depends(client_key),
) -> dict:
    """Rough token estimate (~4 chars/token) — upstream exposes no counter."""
    state = get_state(request)
    messages = body.get("messages")
    if not (isinstance(messages, list) and messages
            and all(isinstance(message, dict) for message in messages)):
        return _error(400, "'messages' must be a non-empty list of message objects")
    try:
        payload = request_to_openai(body, state.registry.resolve(body.get("model"),
                                                                 "anthropic"))
    except (TypeError, ValueError, AttributeError) as exc:
        return _error(400, str(exc))
    chars = 0
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if isinstance(text, str):
                        chars += len(text)
    for tool in payload.get("tools", []) or []:
        chars += len(json.dumps(tool))
    return {"input_tokens": max(1, chars // 4)}
