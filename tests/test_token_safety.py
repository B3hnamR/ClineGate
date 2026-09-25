"""Token refresh must not manage pool state, and forced refreshes dedupe.

Two review findings:

1. `_apply` unconditionally set READY and cleared error_count/last_error, so a
   routine hourly refresh silently un-cooled edge-blocked / 5xx-cooled accounts
   (and could revive exhausted ones). Refreshing a token says nothing about the
   account's health; the pool owns those transitions.
2. `refresh(force=True)` skipped the in-lock freshness re-check, so concurrent
   401s on the same account produced one sequential refresh POST per request,
   all with the same refresh token — a rotation/lockout hazard.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cline_gateway.config import Config
from cline_gateway.pool import Account, AccountState, PoolManager
from cline_gateway.tokens import TokenManager

REFRESH_RESPONSE = {
    "data": {"accessToken": "eyJNEW.TOKEN", "tokenType": "Bearer",
             "expiresAt": "2099-01-01T00:00:00Z",
             "refreshToken": "RNEW"},
    "success": True,
}


class _StubResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _StubClient:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.calls.append(json)
        if self.delay:
            await asyncio.sleep(self.delay)
        return _StubResponse(200, REFRESH_RESPONSE)


def _manager() -> TokenManager:
    cfg = Config()
    pool = PoolManager(cfg.pool, [])
    return TokenManager(cfg, pool)


def _account(**overrides) -> Account:
    base = dict(id="a", access_token="workos:OLD", refresh_token="R",
                expires_at=1)          # expired: refresh has real work to do
    base.update(overrides)
    return Account(**base)


# --------------------------------------------------------------------------- #
# state is the pool's business, not the token manager's
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_refresh_keeps_a_cooling_account_cooling():
    mgr = _manager()
    mgr._client = _StubClient()
    account = _account()
    account.state = AccountState.COOLING
    account.cooldown_until = time.time() + 120
    account.error_count = 2
    account.last_error = "edge_block"

    assert await mgr.refresh(account, force=True) is True

    assert account.access_token.startswith("workos:eyJ")     # token updated
    assert account.state is AccountState.COOLING             # state untouched
    assert account.cooldown_until > time.time()
    assert account.error_count == 2
    assert account.last_error == "edge_block"


@pytest.mark.asyncio
async def test_refresh_does_not_revive_an_exhausted_account():
    mgr = _manager()
    mgr._client = _StubClient()
    account = _account()
    account.state = AccountState.EXHAUSTED
    account.paid_exhausted = True

    assert await mgr.refresh(account, force=True) is True

    assert account.state is AccountState.EXHAUSTED
    assert account.paid_exhausted is True


# --------------------------------------------------------------------------- #
# forced-refresh dedupe
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_forced_refreshes_send_one_post():
    mgr = _manager()
    stub = _StubClient(delay=0.05)
    mgr._client = stub
    account = _account()

    results = await asyncio.gather(
        mgr.refresh(account, force=True),
        mgr.refresh(account, force=True),
    )

    assert results == [True, True]
    assert len(stub.calls) == 1


@pytest.mark.asyncio
async def test_second_forced_refresh_on_a_new_token_still_posts():
    mgr = _manager()
    stub = _StubClient()
    mgr._client = stub
    account = _account()

    assert await mgr.refresh(account, force=True) is True
    # the token changed since the caller's 401 was observed; a *new* 401 with
    # the new token is a different rejection and must refresh again
    assert await mgr.refresh(account, force=True) is True
    assert len(stub.calls) == 2
