"""Regression tests for the 2026-09 bugfix pass.

Each test pins one fixed defect:

* model-scoped errors (429 INFERENCE_CAP_ERROR / 403 ENTITLEMENT_ERROR) must
  rotate to another account instead of failing the request
* a 401 must force a token refresh even when the local expiry looks fine, and
  the same-account retry must hold exactly one in-flight reservation
* max_in_flight_per_account must be a cap, not a preference
* plan-lane models must not be gated by the usage-credit balance
* a recovered balance must restore a retired account
* refresh must persist back to the accounts_dir snapshot and derive an expiry
  from the JWT when the response omits expiresAt
* an interrupted or error-carrying stream must not be recorded as a success
* an Anthropic error frame must not be followed by normal message_stop events
* CRLF-framed SSE must stream incrementally
* /ready must return 503 when no account is usable
* malformed messages must 400 (not 500), and sampling options must reach the
  upstream body
* a malformed snapshot must not take the whole accounts/ folder down
"""

from __future__ import annotations

import base64
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from cline_gateway.config import Config, PoolConfig
from cline_gateway.pool import Account, AccountState, PoolManager
from cline_gateway.registry import Registry
from cline_gateway.service import ChatService, UpstreamFailure
from cline_gateway.store import JsonlCapture, Store
from cline_gateway.tokens import TokenManager, _jwt_exp_ms
from cline_gateway.upstream import UpstreamResponse, build_upstream_body

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resp(status: int, payload: dict | None = None) -> UpstreamResponse:
    raw = (httpx.Response(status, json=payload) if payload is not None
           else httpx.Response(status))
    return UpstreamResponse(status_code=status, headers={}, _resp=raw)


class _PostStub:
    """Stands in for TokenManager's httpx client."""

    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self.response

    async def aclose(self):
        pass


def _service(cfg: Config, pool: PoolManager, client, tokens=None) -> ChatService:
    return ChatService(
        cfg, pool,
        tokens=tokens if tokens is not None else _OkTokens(),
        client=client,
        registry=Registry(),
        store=Store(":memory:"),
        capture=JsonlCapture(".", enabled=False),
    )


class _OkTokens:
    """Refresh always succeeds without touching the network."""

    def __init__(self) -> None:
        self.forced: list[bool] = []

    async def refresh(self, account, force: bool = False) -> bool:
        self.forced.append(force)
        return True

    async def aclose(self) -> None:
        pass


CAP_429 = {
    "error": {"code": "INFERENCE_CAP_ERROR",
              "message": "Error 429: Daily free limit reached on model "
                         "zai/glm-5.3-flash. Try again in 11h 31m"},
}
ENTITLEMENT_403 = {
    "error": {"code": "ENTITLEMENT_ERROR",
              "message": "Error 403: the user is not subscribed to required "
                         "model plan"},
}


class _FailAccountClient:
    """Model-scoped error for account A, 200 for everyone else."""

    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body

    async def send(self, headers, body, stream):
        if "TOKEN-A" in headers.get("authorization", ""):
            return _resp(self.status, self.body)
        return _resp(200)


def _two_account_pool() -> PoolManager:
    return PoolManager(PoolConfig(), [
        Account(id="a1", access_token="workos:TOKEN-A"),
        Account(id="a2", access_token="workos:TOKEN-B"),
    ])


# --------------------------------------------------------------------------- #
# 1. model-scoped errors rotate to another account
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_inference_cap_fails_over_to_the_other_account(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = _two_account_pool()
    svc = _service(cfg, pool, _FailAccountClient(429, CAP_429))

    payload = {"model": "zai/glm-5.3-flash",
               "messages": [{"role": "user", "content": "hi"}]}
    account, _headers, _body, resp, _variant = await svc._open(payload, "default")

    assert account.id == "a2"                       # served by the other account
    assert account.is_model_capped("zai/glm-5.3-flash") is False
    # the capped one is parked, not cooled
    a1 = await pool.find("a1")
    assert a1.is_model_capped("zai/glm-5.3-flash") is True
    assert a1.state is not AccountState.COOLING

    await resp.aclose()
    await pool.release(account)


@pytest.mark.asyncio
async def test_entitlement_fails_over_to_the_other_account(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = _two_account_pool()
    svc = _service(cfg, pool, _FailAccountClient(403, ENTITLEMENT_403))

    payload = {"model": "cline-pass/glm-5.3",
               "messages": [{"role": "user", "content": "hi"}]}
    account, _headers, _body, resp, _variant = await svc._open(payload, "default")

    assert account.id == "a2"
    assert (await pool.find("a1")).is_model_capped("cline-pass/glm-5.3") is True

    await resp.aclose()
    await pool.release(account)


@pytest.mark.asyncio
async def test_cap_on_all_accounts_raises_with_the_parked_reason(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")

    class AllCapped:
        async def send(self, headers, body, stream):
            return _resp(429, CAP_429)

    pool = _two_account_pool()
    svc = _service(cfg, pool, AllCapped())

    payload = {"model": "zai/glm-5.3-flash",
               "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(UpstreamFailure) as exc_info:
        await svc._open(payload, "default")

    assert exc_info.value.code == "INFERENCE_CAP_ERROR"
    assert exc_info.value.release_in_s and exc_info.value.release_in_s > 0


@pytest.mark.asyncio
async def test_auto_free_fallback_tries_next_model_after_all_accounts_cap(tmp_path):
    cfg = Config()
    cfg.models.auto_free_fallback.enabled = True
    cfg.models.auto_free_fallback.chain = ["cline-free/first", "cline-free/second"]

    class ModelClient:
        def __init__(self):
            self.models = []

        async def send(self, headers, body, stream):
            self.models.append(body["model"])
            if body["model"] == "cline-free/first":
                return _resp(429, CAP_429)
            return _resp(200)

    client = ModelClient()
    pool = _two_account_pool()
    svc = _service(cfg, pool, client)
    payload = {"model": "cline-free/first",
               "messages": [{"role": "user", "content": "hi"}]}
    account, _headers, body, resp, _variant = await svc._open(payload, "default")

    assert body["model"] == "cline-free/second"
    assert client.models == ["cline-free/first", "cline-free/first", "cline-free/second"]
    await resp.aclose()
    await pool.release(account)


@pytest.mark.asyncio
async def test_auto_free_fallback_preserves_partial_cap_on_selected_model(tmp_path):
    cfg = Config()
    cfg.models.auto_free_fallback.enabled = True
    cfg.models.auto_free_fallback.chain = ["cline-free/first", "cline-free/second"]
    pool = _two_account_pool()
    svc = _service(cfg, pool, _FailAccountClient(429, CAP_429))
    payload = {"model": "cline-free/first",
               "messages": [{"role": "user", "content": "hi"}]}
    account, _headers, body, resp, _variant = await svc._open(payload, "default")

    assert body["model"] == "cline-free/first"
    assert account.id == "a2"
    await resp.aclose()
    await pool.release(account)


@pytest.mark.asyncio
async def test_auto_free_fallback_returns_structured_exhaustion(tmp_path):
    cfg = Config()
    cfg.models.auto_free_fallback.enabled = True
    cfg.models.auto_free_fallback.chain = ["cline-free/first", "cline-free/second"]

    class AllCapped:
        async def send(self, headers, body, stream):
            return _resp(429, CAP_429)

    svc = _service(cfg, _two_account_pool(), AllCapped())
    payload = {"model": "cline-free/first",
               "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(UpstreamFailure) as exc_info:
        await svc._open(payload, "default")

    exc = exc_info.value
    assert exc.code == "AUTO_FREE_MODELS_EXHAUSTED"
    assert exc.extra()["attempted_models"] == ["cline-free/first", "cline-free/second"]


# --------------------------------------------------------------------------- #
# 2. 401 forces a refresh and holds one reservation
# --------------------------------------------------------------------------- #


class _RetryAfter401Client:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, headers, body, stream):
        self.calls += 1
        if self.calls == 1:
            return _resp(401, {"error": {"code": "UNAUTHORIZED",
                                         "message": "expired"}})
        return _resp(200)


@pytest.mark.asyncio
async def test_401_forces_refresh_and_holds_one_reservation(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    # locally-fresh token: without force=True the refresh would be skipped and
    # the rejected token resent verbatim
    account = Account(id="a1", access_token="workos:STALE",
                      refresh_token="R",
                      expires_at=int(time.time() * 1000) + 3_600_000)
    pool = PoolManager(PoolConfig(), [account])
    tokens = _OkTokens()
    svc = _service(cfg, pool, _RetryAfter401Client(), tokens=tokens)

    payload = {"model": "zai/glm-5.3-flash",
               "messages": [{"role": "user", "content": "hi"}]}
    got, _headers, _body, resp, _variant = await svc._open(payload, "default")

    assert got is account
    assert tokens.forced == [True]           # force=True, not a no-op refresh
    assert account.in_flight == 1           # exactly one reservation held ...
    await pool.release(account)
    assert account.in_flight == 0           # ... released exactly once

    await resp.aclose()


# --------------------------------------------------------------------------- #
# 3. max_in_flight_per_account is a cap
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_in_flight_cap_is_enforced():
    pool = PoolManager(
        PoolConfig(max_in_flight_per_account=2),
        [Account(id="a1", access_token="t")])

    assert await pool.acquire(wait_seconds=0.0) is not None
    assert await pool.acquire(wait_seconds=0.0) is not None
    # third request: no fallback to the saturated account
    assert await pool.acquire(wait_seconds=0.0) is None

    reason = await pool.unavailable_reason("any/model")
    assert reason is not None
    assert reason["code"] == "ACCOUNT_BUSY"

    await pool.release(await pool.find("a1"))
    assert await pool.acquire(wait_seconds=0.0) is not None


# --------------------------------------------------------------------------- #
# 4. plan-lane models are not gated by the credit balance
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_model_routed_on_paid_exhausted_account(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    account = Account(id="a1", access_token="workos:TOKEN-A")
    account.paid_exhausted = True           # usage balance spent...
    account.has_plan = True                 # ...but the subscription is active
    pool = PoolManager(PoolConfig(), [account])
    svc = _service(cfg, pool, _FailAccountClient(200, {}))

    payload = {"model": "cline-pass/glm-5.3",
               "messages": [{"role": "user", "content": "hi"}]}
    got, _headers, _body, resp, _variant = await svc._open(payload, "default")

    assert got is account                   # no "no accounts" detour

    await resp.aclose()
    await pool.release(account)


# --------------------------------------------------------------------------- #
# 5. a recovered balance restores the account
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_restore_reopens_the_paid_lane():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.find("a1")

    await pool.retire(account, reason="balance_threshold")
    assert await pool.ready_count(require_paid=True) == 0

    await pool.restore(account)             # balance poller saw it topped up
    assert await pool.ready_count(require_paid=True) == 1


# --------------------------------------------------------------------------- #
# 6. refresh persistence + expiry derivation
# --------------------------------------------------------------------------- #


SNAPSHOT = (
    "# Cline account snapshot\n"
    "account_id:       usr-test\n"
    "email:            t@example.com\n"
    "access_token:     workos:OLD.TOKEN\n"
    "refresh_token:    ROLD\n"
    "expires_at_ms:    1000\n"
)


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(obj).encode()).rstrip(b"=").decode()


def _jwt(exp: int) -> str:
    return f"workos:{_b64({'alg': 'RS256'})}.{_b64({'exp': exp})}.sig"


def test_jwt_exp_ms_parses_the_claim():
    assert _jwt_exp_ms(_jwt(1234)) == 1_234_000
    assert _jwt_exp_ms("not a jwt") is None
    assert _jwt_exp_ms("") is None


@pytest.mark.asyncio
async def test_force_refresh_contacts_the_api_despite_fresh_expiry(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = PoolManager(cfg.pool, [Account(id="a", access_token="t")])
    mgr = TokenManager(cfg, pool)
    account = Account(id="a", access_token="workos:OK", refresh_token="R",
                      expires_at=int(time.time() * 1000) + 3_600_000)

    stub = _PostStub(httpx.Response(200, json={
        "data": {"accessToken": "workos:NEW", "refreshToken": "R",
                 "expiresAt": "2099-01-01T00:00:00Z"}, "success": True}))
    original = mgr._client
    mgr._client = stub
    await original.aclose()

    assert await mgr.refresh(account) is True
    assert stub.calls == 0                 # fresh: early return, no API call

    assert await mgr.refresh(account, force=True) is True
    assert stub.calls == 1                 # forced: refresh actually happened
    assert account.access_token == "workos:NEW"

    await mgr.aclose()


@pytest.mark.asyncio
async def test_refresh_without_expiry_derives_one_from_the_jwt(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = PoolManager(cfg.pool, [Account(id="a", access_token="t")])
    mgr = TokenManager(cfg, pool)
    account = Account(id="a", access_token="workos:OLD", refresh_token="R",
                      expires_at=1)

    stub = _PostStub(httpx.Response(200, json={
        # no expiresAt in the response at all
        "data": {"accessToken": _jwt(2_000_000_123), "refreshToken": "R"},
        "success": True}))
    original = mgr._client
    mgr._client = stub
    await original.aclose()

    ok = await mgr.refresh(account, force=True)
    await mgr.aclose()

    assert ok is True
    assert account.expires_at == 2_000_000_123 * 1000


@pytest.mark.asyncio
async def test_refresh_updates_the_accounts_dir_snapshot(tmp_path):
    (tmp_path / "acc.txt").write_text(SNAPSHOT, encoding="utf-8")

    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.accounts.source = "accounts_dir"
    cfg.accounts.dir = str(tmp_path)
    cfg.accounts.pool_file = str(tmp_path / "pool.json")

    from cline_gateway.accounts_dir import load_from_accounts_dir, parse_snapshot
    pool = PoolManager(cfg.pool, load_from_accounts_dir(tmp_path))
    mgr = TokenManager(cfg, pool)
    account = (await pool.all())[0]

    stub = _PostStub(httpx.Response(200, json={
        "data": {"accessToken": "eyJNEW", "refreshToken": "RNEW",
                 "expiresAt": "2099-01-01T00:00:00Z"}, "success": True}))
    original = mgr._client
    mgr._client = stub
    await original.aclose()

    ok = await mgr.refresh(account, force=True)
    await mgr.aclose()

    assert ok is True
    fields = parse_snapshot(tmp_path / "acc.txt")
    assert fields["access_token"] == "workos:eyJNEW"   # prefix preserved
    assert fields["refresh_token"] == "RNEW"
    assert int(fields["expires_at_ms"]) == account.expires_at


# --------------------------------------------------------------------------- #
# 7. stream error handling
# --------------------------------------------------------------------------- #


class _FakeResp:
    status_code = 200

    def __init__(self, chunks: list[bytes], exc: Exception | None = None) -> None:
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self._exc = exc

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c
        if self._exc:
            raise self._exc

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aclose(self) -> None:
        pass


def _stream_service(tmp_path, pool=None):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    pool = pool or PoolManager(PoolConfig(),
                               [Account(id="a1", access_token="t")])
    return _service(cfg, pool, client=None), pool


async def _consume(gen) -> bytes:
    return b"".join([chunk async for chunk in gen])


@pytest.mark.asyncio
async def test_interrupted_stream_does_not_clear_cooldown(tmp_path):
    svc, pool = _stream_service(tmp_path)
    account = await pool.find("a1")

    await pool.cool(account, seconds=300, reason="edge_block")
    resp = _FakeResp([b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'],
                     httpx.ReadTimeout("read timed out"))

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open

    out = await _consume(svc.stream(
        {"model": "zai/glm-5.3-flash",
         "messages": [{"role": "user", "content": "hi"}]},
        model="zai/glm-5.3-flash", variant="default",
        dialect="openai", client_key="k", anthropic=False))

    assert b"stream_interrupted" in out
    # the failed stream must NOT report success for the account
    assert account.state is AccountState.COOLING
    assert account.error_count >= 1


@pytest.mark.asyncio
async def test_anthropic_error_frame_is_not_followed_by_a_normal_ending(tmp_path):
    svc, pool = _stream_service(tmp_path)
    account = await pool.find("a1")
    resp = _FakeResp([
        b'data: {"error":{"code":"stream_initialization_failed",'
        b'"message":"At least one message is required"}}\n\n',
        b"data: [DONE]\n\n",
    ])

    async def fake_open(*args, **kwargs):
        return account, {}, {}, resp, "default"

    svc._open = fake_open

    out = await _consume(svc.stream(
        {"model": "cline-free/deepseek-v4.1-flash",
         "messages": [{"role": "user", "content": "hi"}]},
        model="cline-free/deepseek-v4.1-flash", variant="default",
        dialect="anthropic", client_key="k", anthropic=True))

    assert b"event: error" in out
    # failed message must not also "complete" normally
    assert b"message_stop" not in out
    assert b"message_delta" not in out


def test_split_sse_frames_handles_crlf():
    from cline_gateway.service import _split_sse_frames

    frames, pending = _split_sse_frames(b"data: 1\r\n\r\ndata: ")
    assert frames == [b"data: 1"]
    assert pending == b"data: "

    frames, pending = _split_sse_frames(pending + b"2\r\n\r\ndata: 3\n\n")
    assert frames == [b"data: 2", b"data: 3"]
    assert pending == b""


# --------------------------------------------------------------------------- #
# 8. /ready returns 503 when nothing is usable
# --------------------------------------------------------------------------- #


def _app(tmp_path, accounts_payload: str):
    from cline_gateway.app import create_app
    cfg = Config()
    cfg.server.require_client_key = False
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "pool.json")
    (tmp_path / "pool.json").write_text(accounts_payload, encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    cfg.logging.capture = False
    return create_app(cfg)


def test_ready_is_503_with_no_accounts(tmp_path):
    with TestClient(_app(tmp_path, '{"accounts": []}')) as client:
        assert client.get("/health").status_code == 200
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False


def test_ready_is_200_with_an_account(tmp_path):
    pool_json = json.dumps({"accounts": [{
        "id": "a1", "access_token": "workos:t", "refresh_token": "r",
        "expires_at": int(time.time() * 1000) + 3_600_000}]})
    with TestClient(_app(tmp_path, pool_json)) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True


def test_malformed_messages_rejected_as_400_not_500(tmp_path):
    with TestClient(_app(tmp_path, '{"accounts": []}')) as client:
        r = client.post("/v1/chat/completions", json={"model": "x",
                                                      "messages": "hi"})
        assert r.status_code == 400
        assert "error" in r.json()

        r = client.post("/v1/messages", json={"model": "x", "max_tokens": 8,
                                              "messages": "hi"})
        assert r.status_code == 400
        assert r.json().get("type") == "error"


# --------------------------------------------------------------------------- #
# 9. upstream body + translation
# --------------------------------------------------------------------------- #


def test_sampling_options_reach_the_upstream_body():
    body = build_upstream_body(
        {"model": "x", "messages": [], "temperature": 0.3, "top_p": 0.9,
         "stop": ["END"]}, "default")
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9
    assert body["stop"] == ["END"]

    bare = build_upstream_body({"model": "x", "messages": []}, "default")
    assert "temperature" not in bare
    assert "stop" not in bare


def test_anthropic_top_k_survives_translation_and_the_upstream_body():
    from cline_gateway.translate_anthropic import request_to_openai
    payload = request_to_openai(
        {"model": "m", "max_tokens": 8, "messages": [],
         "top_k": 40, "temperature": 0.2}, "m")
    assert payload["top_k"] == 40

    body = build_upstream_body(payload, "default")
    assert body["top_k"] == 40                 # not dropped on the floor

    bare = build_upstream_body({"model": "x", "messages": []}, "default")
    assert "top_k" not in bare                 # only sent when requested


def test_explicit_zero_max_tokens_is_forwarded_not_defaulted():
    from cline_gateway.translate_anthropic import request_to_openai
    payload = request_to_openai(
        {"model": "m", "max_tokens": 0, "messages": []}, "m")
    assert payload["max_tokens"] == 0


def test_non_dict_message_raises_a_protocol_error():
    from cline_gateway.translate_anthropic import messages_to_openai
    with pytest.raises(ValueError):
        messages_to_openai(["hi"])


def test_admin_expiry_parser_accepts_iso_and_rejects_malformed():
    from cline_gateway.api_admin import _admin_expiry
    assert _admin_expiry("2099-01-01T00:00:00Z") > 0
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        _admin_expiry("not-a-date")
    assert exc_info.value.status_code == 400


def test_availability_marks_missing_credentials_unavailable():
    from cline_gateway.availability import UNAVAILABLE, account_model_status
    status = account_model_status(Account(id="missing"), "cline-free/model")
    assert status["status"] == UNAVAILABLE


@pytest.mark.asyncio
async def test_count_tokens_rejects_malformed_messages(tmp_path):
    from cline_gateway.app import create_app
    cfg = Config()
    cfg.server.require_client_key = False
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "pool.json")
    (tmp_path / "pool.json").write_text('{"accounts": []}', encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture = False
    with TestClient(create_app(cfg)) as client:
        response = client.post("/v1/messages/count_tokens",
                               json={"model": "m", "messages": ["bad"]})
    assert response.status_code == 400
    assert response.json()["type"] == "error"


@pytest.mark.asyncio
async def test_count_tokens_ignores_malformed_blocks_without_losing_code_text(tmp_path):
    from cline_gateway.app import create_app
    cfg = Config()
    cfg.server.require_client_key = False
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "pool.json")
    (tmp_path / "pool.json").write_text('{"accounts": []}', encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture = False
    with TestClient(create_app(cfg)) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "cline-free/deepseek-v4.1-flash", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "def answer():\n    return 42"},
                {"type": "text", "text": 123},
            ]}]},
        )
    assert response.status_code == 200
    assert response.json()["input_tokens"] >= 6


async def _chat_chunks():
    yield b'data: {"id":"c1","created":1,"model":"m","choices":' \
          b'[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}\n\n'
    yield b'data: {"id":"c1","created":1,"model":"m","choices":' \
          b'[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}],'
    yield b'"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
    yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_legacy_completion_stream_reshapes_chunks():
    from cline_gateway.api_openai import _legacy_completion_stream
    out = (await _consume(_legacy_completion_stream(_chat_chunks(), "m"))
           ).decode("utf-8")

    assert '"object": "text_completion"' in out
    assert '"text": "he"' in out
    assert '"text": "llo"' in out
    assert '"usage"' in out                      # usage forwarded too
    assert '"delta"' not in out                  # chat shape is gone
    assert out.rstrip().endswith("data: [DONE]")


# --------------------------------------------------------------------------- #
# 10. malformed snapshots do not break the folder
# --------------------------------------------------------------------------- #


def test_malformed_snapshot_does_not_take_the_folder_down(tmp_path):
    from cline_gateway.accounts_dir import load_from_accounts_dir
    (tmp_path / "good.txt").write_text(SNAPSHOT, encoding="utf-8")
    (tmp_path / "bad.txt").write_text(
        "account_id: usr-bad\naccess_token: tok\nexpires_at_ms: nope\n",
        encoding="utf-8")

    accounts = load_from_accounts_dir(tmp_path)
    assert [a.id for a in accounts] == ["usr-test"]
