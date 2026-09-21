"""Edge blocks and interrupted streams — captured 2026-09-16.

Two failure shapes that are neither the account's fault nor the API's:

  1. A global 403 served as HTML by Google's front end, for every model:
       <h2>Your client does not have permission to get URL
           <code>/api/v1/chat/completions</code> from this server.</h2>
     The Cline client surfaced it as "The operation timed out." — the request was
     blocked/aborted before a usable response came back.

  2. A stream that dies after bytes were already sent.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cline_gateway.config import PoolConfig
from cline_gateway.pool import Account, PoolManager
from cline_gateway.registry import Registry
from cline_gateway.service import ChatService
from cline_gateway.store import JsonlCapture, Store
from cline_gateway.upstream import (
    ErrorKind,
    classify,
    edge_block_message,
)

EDGE_403_HTML = (
    '<html><head><meta http-equiv="content-type" content="text/html;charset=utf-8">'
    "<title>403 Forbidden</title></head><body text=#000000 bgcolor=#ffffff>"
    "<h1>Error: Forbidden</h1>"
    "<h2>Your client does not have permission to get URL "
    "<code>/api/v1/chat/completions</code> from this server.</h2>"
    "</body></html>"
)

ENTITLEMENT_403 = ('{"error":{"code":"ENTITLEMENT_ERROR","message":"Error 403: the user '
                   'is not subscribed to required model plan"}}')


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def test_edge_403_classified_as_edge_block():
    assert classify(403, EDGE_403_HTML) is ErrorKind.EDGE_BLOCK


def test_edge_block_is_not_entitlement_and_not_plain_client():
    assert classify(403, EDGE_403_HTML) is not ErrorKind.ENTITLEMENT
    assert classify(403, ENTITLEMENT_403) is ErrorKind.ENTITLEMENT


def test_json_403_stays_json_classified():
    """An HTML check must not swallow ordinary JSON 403s."""
    assert classify(403, '{"error":{"code":"FORBIDDEN"}}') is ErrorKind.CLIENT
    assert classify(403, '{"error":"nope"}') is ErrorKind.CLIENT


def test_edge_block_message_is_readable():
    msg = edge_block_message(EDGE_403_HTML)
    assert "403" in msg
    assert "permission" in msg
    # never returns raw HTML to a client
    assert "<html" not in msg
    assert edge_block_message(b"<html>403 Forbidden</html>") == (
        "upstream edge rejected the request (403 Forbidden)")


def test_404_html_is_not_an_edge_block():
    assert classify(404, EDGE_403_HTML) is ErrorKind.CLIENT


# --------------------------------------------------------------------------- #
# interrupted stream
# --------------------------------------------------------------------------- #


class _FakeResp:
    status_code = 200

    def __init__(self, chunks: list[bytes], exc: Exception | None = None) -> None:
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self._exc = exc
        self.closed = False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c
        if self._exc:
            raise self._exc

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aclose(self) -> None:
        self.closed = True


def _service(tmp_path) -> ChatService:
    cfg = __import__("cline_gateway.config", fromlist=["Config"]).Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    store = Store(cfg.store.sqlite_path)
    capture = JsonlCapture(cfg.logging.capture_dir, enabled=True)
    return ChatService(cfg, pool, tokens=None, client=None,
                       registry=Registry(), store=store, capture=capture)


@pytest.mark.asyncio
async def test_openai_stream_interruption_emits_error_and_done(tmp_path):
    svc = _service(tmp_path)
    account = Account(id="a1", access_token="t")
    first = b'data: {"id":"x","choices":[{"delta":{"content":"partial"}}]}\n\n'
    resp = _FakeResp([first], httpx.ReadTimeout("read timed out"))

    async def fake_open(payload, variant, **kw):
        return account, {}, {}, resp, variant

    svc._open = fake_open                                     # type: ignore[assignment]

    out = b"".join([c async for c in svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="openai", client_key="k", anthropic=False)])

    assert b"partial" in out                                  # what was sent, sent
    assert b"stream_interrupted" in out                       # then a real error
    assert out.rstrip().endswith(b"data: [DONE]")             # and a clean end
    assert resp.closed


@pytest.mark.asyncio
async def test_anthropic_stream_interruption_emits_error_event(tmp_path):
    svc = _service(tmp_path)
    account = Account(id="a1", access_token="t")
    first = (b'data: {"choices":[{"delta":{"content":"partial"},'
             b'"finish_reason":null}]}\n\n')
    resp = _FakeResp([first], httpx.ReadTimeout("read timed out"))

    async def fake_open(payload, variant, **kw):
        return account, {}, {}, resp, variant

    svc._open = fake_open                                     # type: ignore[assignment]

    out = b"".join([c async for c in svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="anthropic", client_key="k", anthropic=True)])

    text = out.decode()
    assert "event: message_start" in text
    assert "partial" in text
    assert "event: error" in text
    assert "stream interrupted" in text.lower()
    assert resp.closed


@pytest.mark.asyncio
async def test_clean_stream_has_no_error_frame(tmp_path):
    svc = _service(tmp_path)
    account = Account(id="a1", access_token="t")
    body = (b'data: {"choices":[{"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n")
    resp = _FakeResp([body], None)

    async def fake_open(payload, variant, **kw):
        return account, {}, {}, resp, variant

    svc._open = fake_open                                     # type: ignore[assignment]

    out = b"".join([c async for c in svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="openai", client_key="k", anthropic=False)])

    assert b"stream_interrupted" not in out
    assert out == body


# --------------------------------------------------------------------------- #
# unavailable_reason: say WHY, not just "no accounts"
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reason_reports_cooling_with_a_retry_hint():
    pool = PoolManager(PoolConfig(cooldown_seconds=42),
                       [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cool(account, reason="edge_block")

    reason = await pool.unavailable_reason("any/model")
    assert reason is not None
    assert reason["code"] == "ACCOUNT_COOLING"
    assert reason["status"] == 503
    assert "edge_block" in reason["message"]
    assert 0 < reason["release_in_s"] <= 42


@pytest.mark.asyncio
async def test_reason_prefers_the_parked_model_over_cooling():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "cline-free/x", 3600,
                         reason="INFERENCE_CAP_ERROR", message="capped")
    await pool.cool(account, reason="edge_block")

    reason = await pool.unavailable_reason("cline-free/x")
    assert reason["code"] == "INFERENCE_CAP_ERROR"      # specific beats generic
    assert reason["status"] == 429


@pytest.mark.asyncio
async def test_reason_ignores_unrelated_models():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "cline-free/x", 3600, reason="INFERENCE_CAP_ERROR")
    await pool.release(account)
    assert await pool.unavailable_reason("some/other") is None


@pytest.mark.asyncio
async def test_reason_reports_no_usable_accounts_when_dead():
    a = Account(id="a1", access_token="t")
    pool = PoolManager(PoolConfig(), [a])
    await pool.kill(a, reason="refresh_failed")
    reason = await pool.unavailable_reason("any")
    assert reason["code"] == "NO_USABLE_ACCOUNTS"


# --------------------------------------------------------------------------- #
# an error frame inside an HTTP 200 stream
# --------------------------------------------------------------------------- #

# captured 2026-09-16: upstream answered 200, then sent this as an SSE frame
def _stream_init_error_frame() -> str:
    """Build the captured error frame with json.dumps so the escaping is valid.

    The upstream message itself embeds a JSON error blob, so hand-writing the
    escaping is a trap - build it.
    """

    inner = json.dumps({
        "error": {"message": "At least one message is required",
                  "type": "invalid_request_error",
                  "param": "messages",
                  "code": "invalid_request_error"},
    }, separators=(",", ":"))
    message = ("Failed to create stream: inference request failed: failed to "
               "generate stream from Vercel: failed to invoke model "
               "'deepseek/deepseek-v4.1-flash' with streaming: request failed "
               "with status 400: " + inner)
    frame = {"error": {"code": "stream_initialization_failed",
                       "message": message,
                       "request_id": "abc123",
                       "type": "stream_error"}}
    return f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n"


STREAM_INIT_ERROR = _stream_init_error_frame()


def test_extract_stream_error_finds_the_frame():
    from cline_gateway.translate_anthropic import extract_stream_error
    err = extract_stream_error(STREAM_INIT_ERROR)
    assert err is not None
    assert err["code"] == "stream_initialization_failed"
    assert "At least one message is required" in err["message"]
    assert err["request_id"] == "abc123"


def test_extract_stream_error_none_for_clean_stream():
    from cline_gateway.translate_anthropic import extract_stream_error
    clean = ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
             "data: [DONE]\n\n")
    assert extract_stream_error(clean) is None


def test_stream_had_content_distinguishes_error_only_streams():
    from cline_gateway.translate_anthropic import stream_had_content
    assert stream_had_content(STREAM_INIT_ERROR) is False
    assert stream_had_content(
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n') is True
    # a tool call counts too
    assert stream_had_content(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}\n\n') is True


@pytest.mark.asyncio
async def test_anthropic_stream_emits_error_event_for_upstream_error_frame(tmp_path):
    """Previously the error frame was dropped and the client saw a truncated stream."""
    from cline_gateway.service import ChatService
    from cline_gateway.store import JsonlCapture, Store
    from cline_gateway.config import Config
    from cline_gateway.registry import Registry

    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    svc = ChatService(cfg, pool, tokens=None, client=None, registry=Registry(),
                      store=Store(cfg.store.sqlite_path),
                      capture=JsonlCapture(cfg.logging.capture_dir, enabled=True))

    account = Account(id="a1", access_token="t")
    resp = _FakeResp([STREAM_INIT_ERROR.encode()], None)

    async def fake_open(payload, variant, **kw):
        return account, {}, {}, resp, variant

    svc._open = fake_open

    out = b"".join([c async for c in svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="anthropic", client_key="k", anthropic=True)])

    text = out.decode()
    assert "event: error" in text
    assert "stream_initialization_failed" in text
    assert "At least one message is required" in text


# --------------------------------------------------------------------------- #
# provider refusals (captured 2026-09-16: claude-opus-5, "violative cyber content")
# --------------------------------------------------------------------------- #


def _refusal_stream() -> str:
    refusal = ("This request triggered restrictions on violative cyber content and "
               "was blocked under Anthropic Usage Policy.")
    first = {"id": "gen-x", "object": "chat.completion.chunk", "created": 1,
             "model": "anthropic/claude-opus-5",
             "choices": [{"index": 0,
                          "delta": {"content": "", "role": "assistant",
                                    "refusal": refusal},
                          "finish_reason": None}]}
    second = {"id": "gen-x",
              "choices": [{"index": 0, "delta": {"content": "", "role": "assistant"},
                           "finish_reason": "content_filter",
                           "native_finish_reason": "refusal"}]}
    return (f"data: {json.dumps(first)}\n\n"
            f"data: {json.dumps(second)}\n\ndata: [DONE]\n\n")


def test_refusal_is_preserved_in_aggregation():
    from cline_gateway.service import aggregate_openai_stream
    agg = aggregate_openai_stream(_refusal_stream(), "anthropic/claude-opus-5", "x")
    choice = agg["choices"][0]
    assert choice["finish_reason"] == "content_filter"
    assert "violative cyber content" in choice["message"]["refusal"]


def test_refusal_maps_to_anthropic_stop_reason_and_text():
    from cline_gateway.service import aggregate_openai_stream
    from cline_gateway.translate_anthropic import response_to_anthropic
    agg = aggregate_openai_stream(_refusal_stream(), "m", "x")
    out = response_to_anthropic(agg, "m")
    assert out["stop_reason"] == "refusal"
    texts = [c["text"] for c in out["content"] if c["type"] == "text"]
    assert any("violative cyber content" in t for t in texts)


def test_anthropic_stream_translates_refusal_to_text():
    from cline_gateway.translate_anthropic import AnthropicStreamTranslator
    tr = AnthropicStreamTranslator("m", "msg_r")
    first = {"choices": [{"delta": {"refusal": "blocked for policy"},
                          "finish_reason": None}]}
    second = {"choices": [{"delta": {}, "finish_reason": "content_filter"}]}
    out = b"".join(tr.feed(first) + tr.feed(second) + tr.finish()).decode()
    assert "blocked for policy" in out
    assert '"stop_reason": "refusal"' in out


def test_api_modules_import_cleanly():
    """Guards a real slip: a handler used a helper whose import was never added,
    so the endpoint 500'd with NameError only at request time."""
    import importlib
    for name in ("cline_gateway.api_openai", "cline_gateway.api_anthropic"):
        mod = importlib.import_module(name)
        src = open(mod.__file__, encoding="utf-8").read()
        for helper in ("extract_stream_error", "stream_had_content"):
            if helper in src:
                assert hasattr(mod, helper), f"{name} uses {helper} but did not import it"
