"""Failover classification tests — driven by the captured 402 body."""

from __future__ import annotations

import pytest

from cline_gateway.upstream import (
    ErrorKind,
    anthropic_error,
    classify,
    openai_error,
    parse_upstream_error,
)

CAPTURED_402 = (
    '{"error":{"code":"insufficient_credits",'
    '"message":"Insufficient balance. Your Cline Credits balance is $-0.00",'
    '"current_balance":-0.001615,"total_spent":0,"total_promotions":0,'
    '"buy_credits_url":"https://app.cline.bot/credits"}}'
)


def test_classify_status_codes():
    assert classify(200) is ErrorKind.OK
    assert classify(401) is ErrorKind.UNAUTHORIZED
    # 403 is ambiguous: without an entitlement code it is a plain client error.
    # ENTITLEMENT_ERROR is a model-scoped condition — see test_model_scoped_errors.py
    assert classify(403) is ErrorKind.CLIENT
    assert classify(403, '{"error":{"code":"ENTITLEMENT_ERROR"}}') is ErrorKind.ENTITLEMENT
    assert classify(402) is ErrorKind.INSUFFICIENT_CREDITS
    assert classify(429) is ErrorKind.RATE_LIMITED
    assert classify(500) is ErrorKind.SERVER
    assert classify(503) is ErrorKind.SERVER
    assert classify(400) is ErrorKind.CLIENT
    assert classify(413) is ErrorKind.CLIENT


def test_parse_captured_402():
    info = parse_upstream_error(402, CAPTURED_402)
    assert info["code"] == "insufficient_credits"
    assert "Insufficient balance" in info["message"]
    assert info["raw"]["error"]["current_balance"] == -0.001615


def test_parse_non_json_error():
    info = parse_upstream_error(502, "<html>bad gateway</html>")
    assert info["code"] == "http_502"
    assert "bad gateway" in info["message"]


def test_openai_error_envelope():
    err = openai_error(402, "insufficient_credits", "no credits")
    assert err["error"]["type"] == "insufficient_quota"
    assert err["error"]["code"] == "insufficient_credits"


def test_anthropic_error_envelope():
    err = anthropic_error(429, "rate_limited", "slow down")
    assert err["type"] == "error"
    assert err["error"]["type"] == "rate_limit_error"

    overloaded = anthropic_error(529, "x", "y")
    assert overloaded["error"]["type"] == "overloaded_error"


@pytest.mark.asyncio
async def test_paid_lane_retires_but_free_lane_survives():
    """A 402 on a paid model must not disable credit-free models on the same account."""
    from cline_gateway.config import PoolConfig
    from cline_gateway.pool import Account, AccountState, PoolManager

    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    assert account is not None

    await pool.retire(account, reason="insufficient_credits", paid_only=True)

    assert account.paid_exhausted is True
    assert account.state is AccountState.READY        # account itself is fine

    # free lane still available
    assert await pool.ready_count(require_paid=False) == 1
    assert await pool.acquire(require_paid=False) is not None

    # paid lane closed
    assert await pool.ready_count(require_paid=True) == 0
    assert await pool.acquire(require_paid=True) is None

    # re-enabling restores the paid lane
    await pool.enable("a1")
    assert await pool.ready_count(require_paid=True) == 1


@pytest.mark.asyncio
async def test_hard_retire_removes_account_entirely():
    from cline_gateway.config import PoolConfig
    from cline_gateway.pool import Account, AccountState, PoolManager

    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.retire(account, reason="dead", paid_only=False)

    assert account.state is AccountState.EXHAUSTED
    assert await pool.ready_count() == 0
    assert await pool.acquire() is None


@pytest.mark.asyncio
async def test_pool_rotates_across_accounts():
    from cline_gateway.config import PoolConfig
    from cline_gateway.pool import Account, PoolManager

    accounts = [Account(id=f"a{i}", access_token="t") for i in range(3)]
    pool = PoolManager(PoolConfig(strategy="least_in_flight"), accounts)

    picked = []
    for _ in range(6):
        acc = await pool.acquire()
        picked.append(acc.id)
        await pool.release(acc)

    # least_in_flight distributes evenly
    assert set(picked) == {"a0", "a1", "a2"}
    assert max(picked.count(a) for a in set(picked)) - min(
        picked.count(a) for a in set(picked)) <= 1


@pytest.mark.asyncio
async def test_quota_aware_excludes_broke_account():
    from cline_gateway.config import PoolConfig
    from cline_gateway.pool import Account, PoolManager

    rich = Account(id="rich", access_token="t", balance_micro=1_000_000)
    broke = Account(id="broke", access_token="t", balance_micro=-1615)
    pool = PoolManager(PoolConfig(strategy="quota_aware", min_balance_micro=0),
                       [rich, broke])

    picks = {await pool.acquire(require_paid=True) for _ in range(4)}
    assert all(a.id == "rich" for a in picks if a)
