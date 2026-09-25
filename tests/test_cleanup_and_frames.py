"""Regression tests for reservation/cleanup safety and in-stream error frames.

Three real leaks found in review:

1. `finally: await resp.aclose(); await pool.release(account)` — if aclose
   raises (or the task is cancelled), the account slot is never released and
   the pool silently loses capacity until restart.
2. A non-HTTP error during `aread()` escaped `complete()` as a generic 500
   instead of a dialect-shaped 502.
3. The OpenAI streaming branch passed an upstream error frame through as if
   it were success (`on_success` cleared the account's error state); the
   non-stream path booked the same error-frame 200 as a success too.
"""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import pytest

from cline_gateway.api_openai import _legacy_completion_stream
from cline_gateway.config import Config, PoolConfig
from cline_gateway.pool import Account, PoolManager
from cline_gateway.registry import Registry
from cline_gateway.service import ChatService, UpstreamFailure
from cline_gateway.store import JsonlCapture, Store

ERROR_FRAME = (
    b'data: {"error":{"code":"stream_initialization_failed",'
    b'"message":"At least one message is required"}}\n\n'
    b"data: [DONE]\n\n"
)
CLEAN_FRAME = (
    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


class _Resp:
    status_code = 200

    def __init__(self, chunks: list[bytes], *, aclose_exc: Exception | None = None,
                 aread_exc: Exception | None = None) -> None:
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self._aclose_exc = aclose_exc
        self._aread_exc = aread_exc
        self.closed = False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def aread(self) -> bytes:
        if self._aread_exc:
            raise self._aread_exc
        return b"".join(self._chunks)

    async def aclose(self) -> None:
        self.closed = True
        if self._aclose_exc:
            raise self._aclose_exc


def _service(tmp_path) -> tuple[ChatService, PoolManager, Account]:
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    account = Account(id="a1", access_token="t")
    pool = PoolManager(PoolConfig(), [account])
    svc = ChatService(cfg, pool, tokens=None, client=None, registry=Registry(),
                      store=Store(cfg.store.sqlite_path),
                      capture=JsonlCapture(cfg.logging.capture_dir, enabled=False))
    return svc, pool, account


async def _consume(gen) -> bytes:
    return b"".join([chunk async for chunk in gen])


# --------------------------------------------------------------------------- #
# 1. cleanup must survive a close failure and always release the slot
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stream_releases_slot_when_aclose_raises(tmp_path):
    svc, pool, account = _service(tmp_path)
    account = await pool.acquire()                       # slot taken, in_flight=1
    assert account.in_flight == 1
    resp = _Resp([CLEAN_FRAME], aclose_exc=RuntimeError("close blew up"))

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open                                # type: ignore[assignment]

    out = await _consume(svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="openai", client_key="k", anthropic=False))

    assert b"ok" in out                                  # response was delivered
    assert account.in_flight == 0                        # ...and the slot freed


@pytest.mark.asyncio
async def test_complete_releases_slot_when_aclose_raises(tmp_path):
    svc, pool, account = _service(tmp_path)
    account = await pool.acquire()
    resp = _Resp([CLEAN_FRAME], aclose_exc=RuntimeError("close blew up"))

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open                                # type: ignore[assignment]

    result = await svc.complete({"model": "m", "messages": []},
                                model="m", variant="default",
                                dialect="openai", client_key="k")

    assert result.status == 200
    assert account.in_flight == 0


@pytest.mark.asyncio
async def test_complete_read_failure_is_a_dialect_shaped_502(tmp_path):
    svc, pool, account = _service(tmp_path)
    account = await pool.acquire()
    resp = _Resp([], aread_exc=httpx.ReadTimeout("read timed out"))

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open                                # type: ignore[assignment]

    with pytest.raises(UpstreamFailure) as excinfo:
        await svc.complete({"model": "m", "messages": []},
                           model="m", variant="default",
                           dialect="openai", client_key="k")
    assert excinfo.value.status == 502
    assert account.in_flight == 0                        # released either way


# --------------------------------------------------------------------------- #
# 1b. cancellation / unexpected errors while a reservation is held in _open
# --------------------------------------------------------------------------- #


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_cancelled_open_releases_the_slot(tmp_path):
    svc, pool, account = _service(tmp_path)

    async def hang(headers, body):
        await asyncio.sleep(30)

    svc._send = hang                                     # type: ignore[assignment]

    task = asyncio.create_task(svc.complete(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]},
        model="m", variant="default", dialect="openai", client_key="k"))
    await _wait_until(lambda: account.in_flight == 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_until(lambda: account.in_flight == 0)


@pytest.mark.asyncio
async def test_unexpected_send_error_releases_the_slot(tmp_path):
    svc, pool, account = _service(tmp_path)

    async def boom(headers, body):
        raise ValueError("not a transport error")

    svc._send = boom                                     # type: ignore[assignment]

    with pytest.raises(ValueError):
        await svc.complete(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]},
            model="m", variant="default", dialect="openai", client_key="k")
    assert account.in_flight == 0


# --------------------------------------------------------------------------- #
# 2. an OpenAI error frame inside a 200 is not a success, streaming or not
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_openai_error_frame_is_not_recorded_as_success(tmp_path):
    svc, pool, account = _service(tmp_path)
    account = await pool.acquire()
    account.error_count = 2                              # pre-existing failures
    account.last_error = "earlier"
    resp = _Resp([ERROR_FRAME])

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open                                # type: ignore[assignment]

    out = await _consume(svc.stream(
        {"model": "m", "messages": []}, model="m", variant="default",
        dialect="openai", client_key="k", anthropic=False))

    # the error frame itself is still passed through (byte-exact contract)
    assert b"stream_initialization_failed" in out
    # ...but the account must not be marked healthy by it
    assert account.error_count == 2
    assert account.last_error == "earlier"
    assert account.in_flight == 0


@pytest.mark.asyncio
async def test_complete_error_frame_is_not_recorded_as_success(tmp_path):
    svc, pool, account = _service(tmp_path)
    account = await pool.acquire()
    account.error_count = 2                              # pre-existing failures
    account.last_error = "earlier"
    resp = _Resp([ERROR_FRAME])

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open                                # type: ignore[assignment]

    result = await svc.complete({"model": "m", "messages": []},
                                model="m", variant="default",
                                dialect="openai", client_key="k")

    assert result.status == 200                          # upstream said 200...
    assert account.error_count == 2                      # ...but not a success
    assert account.last_error == "earlier"
    assert account.in_flight == 0
    with sqlite3.connect(tmp_path / "g.db") as conn:
        assert conn.execute("SELECT status FROM usage").fetchone()[0] == 502


# --------------------------------------------------------------------------- #
# 3. the legacy /v1/completions shim closes its underlying source
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_legacy_completion_stream_closes_the_source():
    closed = False

    async def source():
        nonlocal closed
        try:
            yield b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        finally:
            closed = True

    gen = _legacy_completion_stream(source(), "m")
    first = await gen.__anext__()
    assert b"hi" in first
    await gen.aclose()
    assert closed is True
