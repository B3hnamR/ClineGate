"""Per-model daily cap (INFERENCE_CAP_ERROR) — captured 2026-09-16.

Raw exchange:
    POST /api/v1/chat/completions   model=zai/glm-5.3-flash
    429  no-retry: true
    {"error":{"code":"INFERENCE_CAP_ERROR","message":"Error 429: Daily free limit
      reached on model zai/glm-5.3-flash. Try again in 11h 31m"}}
"""

from __future__ import annotations

import pytest

from cline_gateway.config import PoolConfig
from cline_gateway.pool import Account, AccountState, PoolManager
from cline_gateway.registry import Registry
from cline_gateway.service import ChatService

CAP_BODY = (
    '{"error":{"code":"INFERENCE_CAP_ERROR","message":"Error 429: Daily free limit '
    'reached on model zai/glm-5.3-flash. Try again in 11h 31m"}}'
)
CAP_MESSAGE = ("Error 429: Daily free limit reached on model zai/glm-5.3-flash. "
               "Try again in 11h 31m")


# --------------------------------------------------------------------------- #
# free-model classification
# --------------------------------------------------------------------------- #


def test_catalog_free_models_are_free():
    """stealth/space-bunny-alpha and stealth/pixel-canary are in the live
    catalogue's free section but have no cline-free/ prefix and no :free
    suffix — they must still count as free."""
    assert Registry.is_free("stealth/space-bunny-alpha") is True
    assert Registry.is_free("stealth/pixel-canary") is True
    assert Registry.is_free("cline-free/gemini-3.8-flash") is True
    assert Registry.is_free("cline-free/mimo-v2.6-flash") is True


def test_removed_free_models_are_no_longer_free():
    """zai/glm-5.3-flash left the free section on 2026-09-24 (now usage-billed):
    a 402 on it must retire the paid lane instead of looking like a daily cap."""
    assert Registry.is_free("zai/glm-5.3-flash") is False
    assert Registry.is_free("z-ai/glm-5.3-flash") is False


def test_free_classification_unchanged_for_other_shapes():
    assert Registry.is_free("cline-free/deepseek-v4.1-flash") is True
    assert Registry.is_free("cline-free/anything") is True
    assert Registry.is_free("poolside/laguna-s-2.1:free") is True
    assert Registry.is_free("anthropic/claude-opus-5") is False
    assert Registry.is_free("openai/gpt-6-astra") is False


# --------------------------------------------------------------------------- #
# cap detection + parsing
# --------------------------------------------------------------------------- #


def test_is_model_cap_by_code():
    assert ChatService._is_model_cap({"code": "INFERENCE_CAP_ERROR"}, True) is True
    assert ChatService._is_model_cap({"code": "inference_cap_error"}, False) is True


def test_is_model_cap_by_no_retry_and_message():
    info = {"code": "RATE_LIMITED", "message": "Daily free limit reached"}
    assert ChatService._is_model_cap(info, True) is True
    # without the no-retry header it is an ordinary rate limit
    assert ChatService._is_model_cap(info, False) is False


def test_is_model_cap_ignores_plain_rate_limit():
    info = {"code": "TOO_MANY_REQUESTS", "message": "slow down"}
    assert ChatService._is_model_cap(info, True) is False


def test_parse_cap_seconds_hours_and_minutes():
    assert ChatService._parse_cap_seconds(CAP_MESSAGE) == 11 * 3600 + 31 * 60


def test_parse_cap_seconds_variants():
    assert ChatService._parse_cap_seconds("Try again in 39m") == 39 * 60
    assert ChatService._parse_cap_seconds("Try again in 2h") == 2 * 3600
    assert ChatService._parse_cap_seconds("Try again in 1h 0m") == 3600
    # unparseable -> 1 hour default
    assert ChatService._parse_cap_seconds("no window here") == 3600.0
    # never returns less than a minute
    assert ChatService._parse_cap_seconds("Try again in 0m") >= 60


# --------------------------------------------------------------------------- #
# pool behaviour: park the MODEL, not the account
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cap_parks_model_without_cooling_account():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    assert account is not None
    await pool.release(account)

    await pool.cap_model(account, "zai/glm-5.3-flash", 3600)

    # the model is parked
    assert account.is_model_capped("zai/glm-5.3-flash") is True
    assert await pool.acquire(model="zai/glm-5.3-flash") is None

    # the account is NOT cooled and other models still route
    assert account.state is AccountState.READY
    assert await pool.ready_count() == 1
    other = await pool.acquire(model="cline-free/deepseek-v4.1-flash")
    assert other is not None


@pytest.mark.asyncio
async def test_cap_expiry_releases_the_model():
    import time

    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "m", 60)
    assert account.is_model_capped("m") is True

    # force the cap into the past rather than sleeping out a real minute
    account.model_caps["m"] = time.time() - 1
    assert account.is_model_capped("m") is False
    assert await pool.acquire(model="m") is not None


@pytest.mark.asyncio
async def test_cap_is_per_account():
    """A model capped on one account must still route on another."""
    a = Account(id="a1", access_token="t")
    b = Account(id="a2", access_token="t")
    pool = PoolManager(PoolConfig(), [a, b])

    await pool.cap_model(a, "zai/glm-5.3-flash", 3600)

    picks = set()
    for _ in range(4):
        acc = await pool.acquire(model="zai/glm-5.3-flash")
        if acc:
            picks.add(acc.id)
            await pool.release(acc)
    assert picks == {"a2"}


@pytest.mark.asyncio
async def test_capped_models_appear_in_snapshot():
    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, "zai/glm-5.3-flash", 600)
    snap = await pool.snapshot()
    assert snap["detail"][0]["capped_models"] == ["zai/glm-5.3-flash"]


# --------------------------------------------------------------------------- #
# the message can name the RAW upstream model, not the requested alias
# --------------------------------------------------------------------------- #

# captured: requested cline-free/muse-spark-1.3-contributor
#           message  meta/muse-spark-1.3-contributor
ALIAS_AND_RAW = (
    '{"error":{"code":"INFERENCE_CAP_ERROR","message":"Error 429: Daily free limit '
    'reached on model meta/muse-spark-1.3-contributor. Try again in 10h 52m"}}'
)


def test_cap_window_parses_for_the_alias_case():
    assert ChatService._parse_cap_seconds(
        "Error 429: Daily free limit reached on model "
        "meta/muse-spark-1.3-contributor. Try again in 10h 52m") == 10 * 3600 + 52 * 60


@pytest.mark.asyncio
async def test_park_keys_on_requested_id_not_the_message_name():
    """The park must key on what the client requested. Keying on the name inside
    the message would miss alias requests and re-probe the upstream every call."""
    requested = "cline-free/muse-spark-1.3-contributor"
    raw_in_message = "meta/muse-spark-1.3-contributor"

    pool = PoolManager(PoolConfig(), [Account(id="a1", access_token="t")])
    account = await pool.acquire()
    await pool.cap_model(account, requested, 10 * 3600 + 52 * 60,
                         reason="INFERENCE_CAP_ERROR",
                         message="Daily free limit reached on " + raw_in_message)

    # parked under the requested id
    assert account.is_model_capped(requested) is True
    assert await pool.acquire(model=requested) is None

    # the raw name from the message is NOT a park key, so requesting it is untouched
    assert account.is_model_capped(raw_in_message) is False

    got = await pool.parked_reason(requested)
    assert got is not None
    assert got["code"] == "INFERENCE_CAP_ERROR"
    assert got["release_in_s"] > 0


# --------------------------------------------------------------------------- #
# billing lanes + per-account/per-model availability
# --------------------------------------------------------------------------- #

from cline_gateway.availability import (          # noqa: E402
    AVAILABLE, CAPPED, NO_CREDIT, NO_ENTITLEMENT, account_model_status,
    build_matrix, model_availability,
)
from cline_gateway.registry import model_lane     # noqa: E402


def test_three_lanes():
    assert model_lane("cline-free/deepseek-v4.1-flash") == "free"
    assert model_lane("stealth/space-bunny-alpha") == "free"   # catalogue-free
    assert model_lane("stealth/pixel-canary") == "free"        # catalogue-free
    assert model_lane("cline-pass/glm-5.3") == "plan"
    assert model_lane("cline-cloud/kimi-k3") == "plan"
    assert model_lane("zai/glm-5.3-flash") == "usage"          # left the free section
    assert model_lane("openai/gpt-6-astra") == "usage"
    assert model_lane("anthropic/claude-opus-5") == "usage"


def test_usage_model_needs_credit():
    a = Account(id="a", access_token="t")
    a.paid_exhausted = True
    st = account_model_status(a, "openai/gpt-6-astra")
    assert st["status"] == NO_CREDIT

    a.paid_exhausted = False
    assert account_model_status(a, "openai/gpt-6-astra")["status"] == AVAILABLE


def test_plan_model_needs_a_plan_not_credit():
    """A broke account and a rich account are equally blocked without a plan."""
    a = Account(id="a", access_token="t")
    a.has_plan = False
    a.paid_exhausted = False                 # has credit, still no plan
    st = account_model_status(a, "cline-pass/glm-5.3")
    assert st["status"] == NO_ENTITLEMENT
    assert "plan" in st["reason"]

    a.has_plan = True
    assert account_model_status(a, "cline-pass/glm-5.3")["status"] == AVAILABLE


def test_free_model_ignores_balance_entirely():
    """The point the user made: a -$ balance still serves free models."""
    a = Account(id="a", access_token="t")
    a.paid_exhausted = True
    a.has_plan = False
    for m in ("cline-free/deepseek-v4.1-flash", "stealth/space-bunny-alpha",
              "stealth/pixel-canary", "cline-free/gemini-3.8-flash"):
        assert account_model_status(a, m)["status"] == AVAILABLE, m
    # a former free id that left the free section is correctly NOT available
    assert account_model_status(a, "zai/glm-5.3-flash")["status"] == NO_CREDIT


def test_free_cap_is_per_model_per_account():
    a = Account(id="a", access_token="t")
    a.paid_exhausted = True
    a.model_caps["cline-free/gemini-3.8-flash"] = __import__("time").time() + 3600
    a.model_cap_info["cline-free/gemini-3.8-flash"] = {"code": "INFERENCE_CAP_ERROR",
                                                       "release_at": "x"}
    assert account_model_status(a, "cline-free/gemini-3.8-flash")["status"] == CAPPED
    # every other free model still works on the same account
    assert account_model_status(a, "cline-free/deepseek-v4.1-flash")["status"] == AVAILABLE


def test_availability_reports_partial_and_next_release():
    import time
    a = Account(id="a", access_token="t")
    b = Account(id="b", access_token="t")
    a.model_caps["m"] = time.time() + 7200
    a.model_cap_info["m"] = {"code": "INFERENCE_CAP_ERROR", "release_at": "x"}

    out = model_availability([a, b], "m")
    assert out["overall"] == "partial"       # b can still serve it
    assert out["available_on"] == 1
    assert 7000 < out["next_release_in_s"] <= 7200


def test_matrix_summary_counts():
    a = Account(id="a", access_token="t")
    a.paid_exhausted = True
    b = Account(id="b", access_token="t")
    m = build_matrix([a, b], ["openai/x", "cline-free/y"])
    s = m["summary"]
    assert s["total_models"] == 2
    assert s["paid_models"] == 1
    assert s["free_models"] == 1
    assert s["fully_available"] == 1        # the free one
    assert s["partially_available"] == 1    # the paid one (a has no credit)
