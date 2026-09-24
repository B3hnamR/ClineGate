"""Model-scoped upstream errors must not disable the account.

Two captured cases (2026-09-16):

  INFERENCE_CAP_ERROR (429)   zai/glm-5.3-flash   "Daily free limit reached"
  ENTITLEMENT_ERROR   (403)   cline-pass/glm-5.3  "not subscribed to required model plan"

Both are about ONE MODEL on an otherwise healthy account.
"""

from __future__ import annotations

import pytest

from cline_gateway.config import PoolConfig
from cline_gateway.pool import Account, AccountState, PoolManager
from cline_gateway.service import ENTITLEMENT_PARK_SECONDS, ChatService
from cline_gateway.upstream import ErrorKind, classify, parse_upstream_error

CAP_429 = ('{"error":{"code":"INFERENCE_CAP_ERROR","message":"Error 429: Daily free '
           'limit reached on model zai/glm-5.3-flash. Try again in 11h 31m"}}')
ENTITLEMENT_403 = ('{"error":{"code":"ENTITLEMENT_ERROR","message":"Error 403: the '
                   'user is not subscribed to required model plan"}}')
PLAIN_401 = ('{"error":"Unauthorized: Please make sure you\'re using the latest '
             'version of Cline and re-authenticate your Cline account."}')


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def test_entitlement_403_is_not_unauthorized():
    """The bug this guards: 403 used to map to UNAUTHORIZED, which makes the
    service refresh the token and then kill the account."""
    assert classify(403, ENTITLEMENT_403) is ErrorKind.ENTITLEMENT
    assert classify(403, ENTITLEMENT_403) is not ErrorKind.UNAUTHORIZED


def test_plain_403_is_a_client_error_not_entitlement():
    assert classify(403, '{"error":{"code":"FORBIDDEN"}}') is ErrorKind.CLIENT


def test_401_still_unauthorized():
    assert classify(401, PLAIN_401) is ErrorKind.UNAUTHORIZED


def test_string_error_body_does_not_crash_classify():
    assert classify(403, '{"error":"nope"}') is ErrorKind.CLIENT
    assert classify(401, '{"error":"nope"}') is ErrorKind.UNAUTHORIZED


def test_entitlement_codes_recognised():
    for code in ("ENTITLEMENT_ERROR", "NOT_SUBSCRIBED", "PLAN_REQUIRED",
                 "MODEL_NOT_IN_PLAN"):
        body = '{"error":{"code":"%s","message":"x"}}' % code
        assert classify(403, body) is ErrorKind.ENTITLEMENT, code


def test_parse_entitlement_body():
    info = parse_upstream_error(403, ENTITLEMENT_403)
    assert info["code"] == "ENTITLEMENT_ERROR"
    assert "not subscribed" in info["message"]


# --------------------------------------------------------------------------- #
# pool: parking behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_entitlement_parks_only_the_model():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "cline-pass/glm-5.3", ENTITLEMENT_PARK_SECONDS,
                         reason="ENTITLEMENT_ERROR")

    assert account.is_model_capped("cline-pass/glm-5.3") is True
    assert account.state is AccountState.READY                 # account untouched
    assert await pool.ready_count() == 1
    assert await pool.acquire(model="cline-pass/glm-5.3") is None
    assert await pool.acquire(model="cline-free/deepseek-v4.1-flash") is not None


@pytest.mark.asyncio
async def test_cap_seconds_default_for_entitlement_message():
    """The entitlement message carries no 'in Xh' window -> 6h park."""
    seconds = ChatService._parse_cap_seconds(
        "Error 403: the user is not subscribed to required model plan")
    assert seconds == 3600.0                                   # parser default
    assert ENTITLEMENT_PARK_SECONDS == 6 * 3600                # service overrides


def test_free_classification_covers_catalog_ids():
    from cline_gateway.registry import Registry
    # live free section 2026-09-24: stealth/ has no cline-free prefix
    assert Registry.is_free("stealth/space-bunny-alpha")
    assert Registry.is_free("cline-free/gemini-3.8-flash")
    # cline-pass models are NOT free (they need a pass/plan)
    assert not Registry.is_free("cline-pass/glm-5.3")
    # zai/glm-5.3-flash left the free section (now usage-billed)
    assert not Registry.is_free("zai/glm-5.3-flash")


@pytest.mark.asyncio
async def test_parked_reason_reports_entitlement():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "cline-pass/glm-5.3", 600,
                         reason="ENTITLEMENT_ERROR",
                         message="Error 403: the user is not subscribed to required model plan")
    got = await pool.parked_reason("cline-pass/glm-5.3")
    assert got is not None
    assert got["status"] == 403
    assert got["code"] == "ENTITLEMENT_ERROR"
    assert "not subscribed" in got["message"]
    assert got["release_at"] is not None          # absolute timer, not just a string
    assert got["release_in_s"] > 0


@pytest.mark.asyncio
async def test_parked_reason_reports_cap_as_429():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "zai/glm-5.3-flash", 600,
                         reason="INFERENCE_CAP_ERROR",
                         message="Error 429: Daily free limit reached")
    got = await pool.parked_reason("zai/glm-5.3-flash")
    assert got["status"] == 429
    assert got["code"] == "INFERENCE_CAP_ERROR"


@pytest.mark.asyncio
async def test_parked_reason_none_for_unparked_model():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    assert await pool.parked_reason("some/model") is None


# --------------------------------------------------------------------------- #
# the release timer
# --------------------------------------------------------------------------- #


def test_cap_seconds_parses_the_captured_window():
    assert ChatService._parse_cap_seconds(CAP_429) == 11 * 3600 + 31 * 60


@pytest.mark.asyncio
async def test_cap_stores_absolute_release_and_relative_window():
    import time

    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "zai/glm-5.3-flash", 11 * 3600 + 31 * 60,
                         reason="INFERENCE_CAP_ERROR",
                         message="Error 429: Daily free limit reached")

    info = account.model_cap_info["zai/glm-5.3-flash"]
    assert info["release_at"] is not None
    assert info["release_in_s"] == 11 * 3600 + 31 * 60
    # the absolute timestamp and the window agree
    assert abs(account.model_caps["zai/glm-5.3-flash"]
               - (time.time() + info["release_in_s"])) < 5


@pytest.mark.asyncio
async def test_snapshot_exposes_model_cap_timers():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "zai/glm-5.3-flash", 3600,
                         reason="INFERENCE_CAP_ERROR", message="cap")
    snap = await pool.snapshot()
    detail = snap["detail"][0]
    assert detail["capped_models"] == ["zai/glm-5.3-flash"]
    cap = detail["model_caps"]["zai/glm-5.3-flash"]
    assert cap["code"] == "INFERENCE_CAP_ERROR"
    assert cap["release_at"] is not None
    assert 0 < cap["release_in_s"] <= 3600


def test_error_envelopes_carry_the_timer():
    from cline_gateway.upstream import anthropic_error, openai_error
    extra = {"release_at": "2026-09-17T01:10:00+00:00", "release_in_s": 41460}
    o = openai_error(429, "INFERENCE_CAP_ERROR", "capped", extra)
    assert o["error"]["release_at"] == "2026-09-17T01:10:00+00:00"
    assert o["error"]["release_in_s"] == 41460
    a = anthropic_error(429, "x", "capped", extra)
    assert a["error"]["release_at"] == "2026-09-17T01:10:00+00:00"
    # no extra -> unchanged shape
    assert "release_at" not in openai_error(429, "x", "y")["error"]
