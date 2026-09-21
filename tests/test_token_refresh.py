"""Token refresh contract tests — fixture is the real captured response."""

from __future__ import annotations

import time

import pytest

from cline_gateway.config import Config
from cline_gateway.pool import Account, PoolManager
from cline_gateway.tokens import TokenManager, _extract_token, _parse_expiry

# exactly what api.cline.bot/api/v1/auth/refresh returned on 2026-09-16
CAPTURED_REFRESH_RESPONSE = {
    "data": {
        "accessToken": "eyJTEST.HEADER.SIG",
        "tokenType": "Bearer",
        "expiresAt": "2026-09-16T02:52:30Z",
        "refreshToken": "TEST-refresh-token-0001",
        "userInfo": {
            "subject": "user_TEST000000000000000000A",
            "clineUserId": "usr-TEST0000000000000000000A",
            "email": "someone@example.com",
            "firstName": "Test",
        },
    },
    "success": True,
}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_extract_captured_shape():
    parsed = _extract_token(CAPTURED_REFRESH_RESPONSE)
    assert parsed is not None
    assert parsed["access_token"].startswith("eyJ")
    assert parsed["refresh_token"] == "TEST-refresh-token-0001"
    assert parsed["expires_at"] == 1789527150000        # ISO -> epoch ms


def test_expiry_iso_string():
    assert _parse_expiry("2026-09-16T02:52:30Z") == 1789527150000


def test_expiry_epoch_seconds_and_millis():
    assert _parse_expiry(1789527150) == 1789527150000
    assert _parse_expiry(1789527150000) == 1789527150000
    assert _parse_expiry("1789527150") == 1789527150000


def test_expiry_garbage():
    assert _parse_expiry("not-a-date") == 0
    assert _parse_expiry(None) == 0


def test_extract_flat_shape_still_works():
    flat = {"accessToken": "abc", "refreshToken": "r", "expiresAt": 1789527150}
    parsed = _extract_token(flat)
    assert parsed["access_token"] == "abc"
    assert parsed["expires_at"] == 1789527150000


def test_extract_rejects_empty():
    assert _extract_token({"data": {}, "success": True}) is None
    assert _extract_token("nope") is None


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def _manager() -> TokenManager:
    cfg = Config()
    pool = PoolManager(cfg.pool, [])
    return TokenManager(cfg, pool)


def test_apply_restores_workos_prefix():
    """providers.json stores 'workos:...'; the refresh response does not."""
    mgr = _manager()
    account = Account(id="a", access_token="workos:OLD.TOKEN",
                      refresh_token="R", expires_at=1)

    mgr._apply(account, _extract_token(CAPTURED_REFRESH_RESPONSE))

    assert account.access_token.startswith("workos:eyJ")
    assert account.refresh_token == "TEST-refresh-token-0001"
    assert account.expires_at == 1789527150000
    # deterministic: expires_in is derived from expires_at, not "in the future"
    # (the captured expiry is a fixed instant and will pass in real time)
    assert account.expires_in() == pytest.approx(1789527150 - time.time(), abs=5)


def test_apply_does_not_double_prefix():
    mgr = _manager()
    account = Account(id="a", access_token="workos:OLD", refresh_token="R")
    mgr._apply(account, {"access_token": "workos:NEW", "refresh_token": "R2",
                         "expires_at": 0})
    assert account.access_token == "workos:NEW"


def test_apply_without_prefix_when_original_had_none():
    mgr = _manager()
    account = Account(id="a", access_token="PLAIN", refresh_token="R")
    mgr._apply(account, {"access_token": "PLAIN2", "refresh_token": "R2",
                         "expires_at": 0})
    assert account.access_token == "PLAIN2"


# --------------------------------------------------------------------------- #
# end-to-end refresh against a stub transport
# --------------------------------------------------------------------------- #


class _StubResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _StubClient:
    def __init__(self, response: _StubResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


@pytest.mark.asyncio
async def test_refresh_sends_captured_contract():
    mgr = _manager()
    stub = _StubClient(_StubResponse(200, CAPTURED_REFRESH_RESPONSE))
    mgr._client = stub                                   # type: ignore[assignment]

    account = Account(id="a", access_token="workos:OLD", refresh_token="OLDREF",
                      expires_at=int(time.time() * 1000) - 1000)   # expired

    assert await mgr.refresh(account) is True

    call = stub.calls[0]
    assert call["url"] == "https://api.cline.bot/api/v1/auth/refresh"
    assert call["json"] == {"refreshToken": "OLDREF",
                            "grantType": "refresh_token"}
    assert call["headers"]["user-agent"] == "Bun/1.3.13"
    assert account.access_token.startswith("workos:eyJ")
    assert account.expires_at == 1789527150000


@pytest.mark.asyncio
async def test_refresh_skipped_when_token_still_fresh():
    mgr = _manager()
    stub = _StubClient(_StubResponse(200, CAPTURED_REFRESH_RESPONSE))
    mgr._client = stub                                   # type: ignore[assignment]

    account = Account(id="a", access_token="workos:OK", refresh_token="R",
                      expires_at=int(time.time() * 1000) + 3_600_000)

    assert await mgr.refresh(account) is True
    assert stub.calls == []                              # no network call


@pytest.mark.asyncio
async def test_refresh_failure_returns_false():
    mgr = _manager()
    stub = _StubClient(_StubResponse(401, {"error": "bad"}))
    mgr._client = stub                                   # type: ignore[assignment]

    account = Account(id="a", access_token="workos:OLD", refresh_token="R",
                      expires_at=1)
    # no source config available either -> False
    assert await mgr.refresh(account) is False
    assert len(stub.calls) == 1
