"""Paid-lane balance lifecycle: poll -> retire -> routing -> recovery.

Exercises the whole chain the GUI and router depend on:

* balance_loop polls /users/{id}/balance and retires at/below threshold
* a retired account is excluded from usage-lane routing but still serves
  free-lane and plan-lane models
* a recovered balance restores the account (no manual /enable needed)
* a 402 from upstream retires the lane and fails over to another account
* fetch_balance parses the captured response shape
* admin reload populates balance/plan for newly added accounts
"""

from __future__ import annotations

import time

import httpx
import pytest

from cline_gateway.config import Config, PoolConfig
from cline_gateway.pool import Account, AccountState, PoolManager
from cline_gateway.registry import Registry, model_lane
from cline_gateway.service import ChatService, UpstreamFailure
from cline_gateway.store import JsonlCapture, Store
from cline_gateway.tokens import TokenManager
from cline_gateway.upstream import UpstreamResponse


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _BalanceStub:
    """Stands in for TokenManager's httpx client; serves balances by account."""

    def __init__(self, balances: dict[str, int | None]) -> None:
        self.balances = balances

    def _respond(self, url: str):
        for aid, bal in self.balances.items():
            if f"/users/{aid}/balance" in url:
                if bal is None:
                    return httpx.Response(500)
                return httpx.Response(200, json={"data": {"balance": bal}})
        return httpx.Response(404)

    async def get(self, url, headers=None):
        return self._respond(url)

    async def aclose(self):
        pass


async def _mgr_with_balances(tmp_path, balances: dict[str, int | None]):
    """TokenManager whose HTTP client serves the given balances."""
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = PoolManager(PoolConfig(min_balance_micro=0),
                       [Account(id=aid, access_token=f"workos:{aid}")
                        for aid in balances])
    mgr = TokenManager(cfg, pool)
    original, mgr._client = mgr._client, _BalanceStub(balances)
    await original.aclose()
    return cfg, pool, mgr


async def _one_balance_sweep(mgr: TokenManager) -> None:
    """Run exactly one production balance sweep (no re-implementation)."""
    await mgr.balance_sweep()


# --------------------------------------------------------------------------- #
# 1. lane classification sanity (the gate the router trusts)
# --------------------------------------------------------------------------- #


def test_lane_classification():
    assert model_lane("cline-free/deepseek-v4.1-flash") == "free"
    assert model_lane("anything:free") == "free"
    assert model_lane("cline-pass/glm-5.3") == "plan"
    assert model_lane("cline-cloud/kimi-k3") == "plan"
    assert model_lane("anthropic/claude-opus-5") == "usage"
    assert model_lane("zai/glm-5.3") == "usage"
    assert model_lane("") == "usage"          # unknown -> usage gate (safe)


# --------------------------------------------------------------------------- #
# 2. balance polling: retire at threshold, keep free+plan lanes
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_balance_poll_retires_usage_lane_only(tmp_path):
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": -1615})
    try:
        assert (await mgr.fetch_balance(await pool.find("a1"))) == -1615

        await _one_balance_sweep(mgr)
        a1 = await pool.find("a1")

        assert a1.balance_micro == -1615            # captured shape parsed
        assert a1.paid_exhausted is True            # usage lane retired
        assert a1.state is AccountState.READY       # account itself healthy

        # usage-lane model: not routable
        acc = await pool.acquire(require_paid=True, model="zai/glm-5.3",
                                 wait_seconds=0.0)
        assert acc is None
        # free-lane model: still routable
        acc = await pool.acquire(require_paid=False,
                                 model="cline-free/deepseek-v4.1-flash",
                                 wait_seconds=0.0)
        assert acc is a1
        await pool.release(acc)
        # plan-lane model: still routable (subscription, not credits)
        acc = await pool.acquire(require_paid=False,
                                 model="cline-pass/glm-5.3",
                                 wait_seconds=0.0)
        assert acc is a1
        await pool.release(acc)
    finally:
        await mgr.aclose()


@pytest.mark.asyncio
async def test_balance_poll_no_retire_above_threshold(tmp_path):
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": 499_670})
    try:
        await _one_balance_sweep(mgr)
        a1 = await pool.find("a1")
        assert a1.paid_exhausted is False
        assert a1.state is AccountState.READY
        acc = await pool.acquire(require_paid=True, model="zai/glm-5.3",
                                 wait_seconds=0.0)
        assert acc is a1
        await pool.release(acc)
    finally:
        await mgr.aclose()


@pytest.mark.asyncio
async def test_failed_balance_fetch_retires_nothing(tmp_path):
    # HTTP 500 / None: no state change, no flapping
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": None})
    try:
        assert await mgr.fetch_balance(await pool.find("a1")) is None
        await _one_balance_sweep(mgr)
        a1 = await pool.find("a1")
        assert a1.paid_exhausted is False
        assert a1.state is AccountState.READY
    finally:
        await mgr.aclose()


# --------------------------------------------------------------------------- #
# 3. recovery: topped-up balance restores the lane automatically
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recovered_balance_restores_usage_lane(tmp_path):
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": -1615})
    try:
        await _one_balance_sweep(mgr)
        a1 = await pool.find("a1")
        assert a1.paid_exhausted is True

        # user tops up; the next poll restores
        mgr._client.balances["a1"] = 250_000
        await _one_balance_sweep(mgr)

        assert a1.paid_exhausted is False
        acc = await pool.acquire(require_paid=True, model="zai/glm-5.3",
                                 wait_seconds=0.0)
        assert acc is a1
        await pool.release(acc)
    finally:
        await mgr.aclose()


@pytest.mark.asyncio
async def test_retire_then_hard_exhausted_also_restores(tmp_path):
    # a full EXHAUSTED retirement (paid_only=False) must also come back
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": 10})
    try:
        await pool.retire(await pool.find("a1"), reason="dead", paid_only=False)
        assert (await pool.find("a1")).state is AccountState.EXHAUSTED

        mgr._client.balances["a1"] = 500_000
        await _one_balance_sweep(mgr)
        a1 = await pool.find("a1")
        assert a1.state is AccountState.READY
        assert a1.paid_exhausted is False
    finally:
        await mgr.aclose()


# --------------------------------------------------------------------------- #
# 4. upstream 402 retires the lane and fails over
# --------------------------------------------------------------------------- #


def _resp(status: int, payload: dict | None = None) -> UpstreamResponse:
    raw = (httpx.Response(status, json=payload) if payload is not None
           else httpx.Response(status))
    return UpstreamResponse(status_code=status, headers={}, _resp=raw)


INSUFFICIENT_402 = {
    "error": {"code": "insufficient_credits",
              "message": "Insufficient balance. Your Cline Credits balance is "
                         "$-0.00",
              "current_balance": -1615},
}


class _RetiringClient:
    """402s for chosen tokens, 200 for everyone else."""

    def __init__(self, retire_tokens: set[str]) -> None:
        self.retire_tokens = retire_tokens

    async def send(self, headers, body, stream):
        auth = headers.get("authorization", "")
        if any(t in auth for t in self.retire_tokens):
            return _resp(402, INSUFFICIENT_402)
        return _resp(200)


@pytest.mark.asyncio
async def test_402_retires_lane_and_fails_over_to_second_account(tmp_path):
    cfg = Config()
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    pool = PoolManager(PoolConfig(), [
        Account(id="a1", access_token="workos:TOKEN-A"),
        Account(id="a2", access_token="workos:TOKEN-B"),
    ])

    class _Tok:
        async def refresh(self, account, force=False):
            return True

        async def aclose(self):
            pass

    client = _RetiringClient({"TOKEN-A"})
    svc = ChatService(cfg, pool, tokens=_Tok(), client=client,
                      registry=Registry(), store=Store(":memory:"),
                      capture=JsonlCapture(".", enabled=False))

    payload = {"model": "zai/glm-5.3",
               "messages": [{"role": "user", "content": "hi"}]}
    account, _h, _b, resp, _v = await svc._open(payload, "default")

    # served by the healthy account; the 402ing one kept its free lane
    assert account.id == "a2"
    a1 = await pool.find("a1")
    assert a1.paid_exhausted is True
    assert a1.state is AccountState.READY
    await resp.aclose()
    await pool.release(account)

    # now BOTH accounts 402: "try anyway" confirms against the upstream and
    # the client sees the real 402, not a generic no-accounts 503
    client.retire_tokens = {"TOKEN-A", "TOKEN-B"}
    with pytest.raises(UpstreamFailure) as exc_info:
        await svc._open(payload, "default")
    assert exc_info.value.status == 402
    assert exc_info.value.code == "insufficient_credits"
    assert "Insufficient balance" in exc_info.value.message


# --------------------------------------------------------------------------- #
# 5. admin reload populates balance for newly added accounts
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_admin_reload_fetches_balance_for_new_accounts(tmp_path):
    """The import path, end to end through the real FastAPI route: a new
    account lands in the pool with its balance already populated."""
    import json as _json

    from fastapi.testclient import TestClient

    from cline_gateway.app import create_app

    pool_file = tmp_path / "pool.json"
    pool_file.write_text('{"accounts": []}', encoding="utf-8")   # empty at start

    cfg = Config()
    cfg.server.require_client_key = False
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(pool_file)
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture = False
    cfg.logging.capture_dir = str(tmp_path / "logs")
    cfg.pool.balance_poll_seconds = 0        # keep the test deterministic

    app = create_app(cfg)

    with TestClient(app) as client:
        # the account appears AFTER startup (the import flow): write it to the
        # pool file, then reload picks it up
        pool_file.write_text(_json.dumps({"accounts": [{
            "id": "fresh", "email": "f@x.com", "access_token": "workos:t",
            "refresh_token": "r",
            "expires_at": int(time.time() * 1000) + 3_600_000,
        }]}), encoding="utf-8")
        # serve balances from the stub instead of the real upstream
        # (app_state exists only once the lifespan has started)
        app.state.app_state.tokens._client = _BalanceStub({"fresh": 499_670})

        r = client.post("/admin/pool/reload",
                        headers={"Authorization": "Bearer gw-admin-change-me"})
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == 1

        # balance was fetched at reload time, visible in the pool state
        s = client.get("/admin/pool/state",
                       headers={"Authorization": "Bearer gw-admin-change-me"})
        detail = s.json()["detail"][0]
        assert detail["id"] == "fresh"
        assert detail["balance_micro"] == 499_670


# --------------------------------------------------------------------------- #
# 6. balance rendering (GUI column) and public snapshot
# --------------------------------------------------------------------------- #


def test_public_snapshot_carries_balance():
    a = Account(id="a1", access_token="t")
    a.balance_micro = -1615
    pub = a.to_public()
    assert pub["balance_micro"] == -1615
    # the GUI renders this as USD: -1615 micro == -$0.001615
    assert a.balance_micro / 1e6 == pytest.approx(-0.001615)


def test_gui_balance_column_formats_micro_usd():
    from cline_gateway.gui import format_balance

    # live polled balance always wins (this is the pool's truth)
    assert format_balance({"balance_micro": -1615, "notes": {}}) == "-0.0016"
    assert format_balance({"balance_micro": 499670, "notes": {}}) == "0.4997"
    # a stale snapshot figure must never override the live value
    assert format_balance({"balance_micro": -1615,
                           "notes": {"balance_usd": "0.499670"}}) == "-0.0016"
    # import-time snapshot figure, clearly marked, when no live value exists
    assert format_balance({"balance_micro": None,
                           "notes": {"balance_usd": "0.499670"}}) == "~0.4997"
    # nothing known at all
    assert format_balance({"balance_micro": None, "notes": {}}) == "?"
    assert format_balance({}) == "?"
    assert format_balance({"notes": {"balance_usd": "junk"}}) == "?"

# --------------------------------------------------------------------------- #
# check_account: immediate settle on every path that learns a balance
# (previously the threshold logic only ran in the 300s sweep, so the manual
# $ button and freshly added accounts showed an optimistic paid lane)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_check_account_settles_paid_lane_immediately(tmp_path):
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": -500})
    try:
        a1 = await pool.find("a1")
        assert a1.paid_exhausted is False
        await mgr.check_account(a1)              # no sweep involved
        assert a1.balance_micro == -500
        assert a1.paid_exhausted is True
        assert a1.state is AccountState.READY
    finally:
        await mgr.aclose()


@pytest.mark.asyncio
async def test_check_account_restores_recovered_lane(tmp_path):
    cfg, pool, mgr = await _mgr_with_balances(tmp_path, {"a1": 1_000_000})
    try:
        a1 = await pool.find("a1")
        a1.paid_exhausted = True                 # as if a prior 402 retired it
        await mgr.check_account(a1)
        assert a1.paid_exhausted is False
    finally:
        await mgr.aclose()


@pytest.mark.asyncio
async def test_balance_endpoint_applies_threshold_not_just_fetches(tmp_path):
    """POST /admin/accounts/{id}/balance must settle the paid lane too."""
    from fastapi.testclient import TestClient

    from cline_gateway.app import create_app
    from cline_gateway.config import load_config

    cfg = load_config()
    cfg.update.enabled = False
    cfg.accounts.source = "accounts_dir"
    cfg.accounts.dir = str(tmp_path / "accounts")
    app = create_app(cfg)
    with TestClient(app) as client:
        state = app.state.app_state
        account = Account(id="b1", access_token="workos:b1")
        await state.pool.upsert(account)
        original = state.tokens._client
        state.tokens._client = _BalanceStub({"b1": -50})
        try:
            r = client.post(f"/admin/accounts/b1/balance",
                            headers={"Authorization": f"Bearer {cfg.server.admin_key}"})
            assert r.status_code == 200
            body = r.json()
            assert body["balance_micro"] == -50
            assert body["paid_exhausted"] is True       # settled now, not later
            assert account.paid_exhausted is True
        finally:
            state.tokens._client = original
