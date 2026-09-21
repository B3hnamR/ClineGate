"""Tests for the completion-gap features: unknown-model probe, RPM limits,
catalogue completeness, and the anthropic cache_control knob."""

from __future__ import annotations

import httpx
import pytest

from cline_gateway.config import ClientKey, Config
from cline_gateway.pool import Account, PoolManager
from cline_gateway.ratelimit import RateLimiter
from cline_gateway.registry import Registry
from cline_gateway.service import ChatService
from cline_gateway.store import JsonlCapture, Store
from cline_gateway.tokens import TokenManager
from cline_gateway.upstream import UpstreamResponse, build_upstream_body


# --------------------------------------------------------------------------- #
# rate limiter
# --------------------------------------------------------------------------- #


def test_rate_limiter_allows_within_rpm():
    rl = RateLimiter()
    for _ in range(5):
        allowed, _ = rl.check("k", rpm=5)
        assert allowed


def test_rate_limiter_blocks_at_limit():
    rl = RateLimiter()
    for _ in range(3):
        allowed, _ = rl.check("k", rpm=3)
        assert allowed
    allowed, retry_after = rl.check("k", rpm=3)
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_zero_means_unlimited():
    rl = RateLimiter()
    for _ in range(1000):
        assert rl.check("k", rpm=0)[0]


def test_rate_limiter_keys_are_independent():
    rl = RateLimiter()
    for _ in range(2):
        rl.check("a", rpm=2)
    assert not rl.check("a", rpm=2)[0]
    assert rl.check("b", rpm=2)[0]


def test_rate_limiter_reset():
    rl = RateLimiter()
    rl.check("a", rpm=1)
    assert not rl.check("a", rpm=1)[0]
    rl.reset("a")
    assert rl.check("a", rpm=1)[0]


def test_client_key_rpm_config():
    assert ClientKey(key="k", name="n", rpm=120).rpm == 120
    assert ClientKey(key="k").rpm == 0


# --------------------------------------------------------------------------- #
# registry completeness
# --------------------------------------------------------------------------- #


def test_cline_pass_family_uses_default_variant():
    reg = Registry()
    for model in ("cline-pass/deepseek-v4-flash", "cline-pass/glm-5.3",
                  "cline-pass/mimo-v2.5-pro"):
        assert reg.variant_for(model) == "default"


def test_cline_cloud_family_uses_default_variant():
    # captured 2026-09-19: the kimi-k3 family moved to the reasoning shape
    # (no token-limit key, reasoning_effort-driven); the rest stay default
    reg = Registry()
    assert reg.variant_for("cline-cloud/kimi-k3") == "reasoning"
    assert reg.variant_for("cline-cloud/deepseek-v4-flash") == "default"
    assert reg.variant_for("cline-cloud/glm-5.2") == "default"


def test_knows_distinguishes_catalogued_from_unknown():
    reg = Registry()
    assert reg.knows("cline-pass/glm-5.2")
    assert reg.knows("anthropic/claude-opus-5")
    assert not reg.knows("newvendor/model-x")


def test_catalogue_includes_catalogued_families():
    cat = {m["id"]: m for m in Registry().catalogue()}
    assert "cline-pass/glm-5.2" in cat
    assert "cline-cloud/kimi-k3" in cat
    assert cat["cline-pass/glm-5.2"]["captured"] is False
    assert cat["anthropic/claude-opus-5"]["captured"] is True


# --------------------------------------------------------------------------- #
# anthropic cache_control knob
# --------------------------------------------------------------------------- #


def test_anthropic_cache_control_can_be_disabled():
    body = build_upstream_body({"model": "x", "messages": []}, "anthropic",
                               anthropic_cache_control=False)
    assert "cache_control" not in body
    assert "max_tokens" in body


def test_anthropic_cache_control_default_keeps_captured_dialect():
    body = build_upstream_body({"model": "x", "messages": []}, "anthropic")
    assert body["cache_control"] == {"type": "ephemeral"}


# --------------------------------------------------------------------------- #
# unknown-model probe
# --------------------------------------------------------------------------- #


def _upstream_response(status: int, payload: dict | None = None) -> UpstreamResponse:
    raw = (httpx.Response(status, json=payload) if payload is not None
           else httpx.Response(status))
    return UpstreamResponse(status_code=status, headers={}, _resp=raw)


class StubClient:
    """400s the default shape, 200s the openai-nextgen shape."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def send(self, headers, body, stream):
        self.bodies.append(body)
        if "max_completion_tokens" in body or body.get("model") != "newvendor/model-x":
            return _upstream_response(200)
        return _upstream_response(400, {
            "error": {"code": "invalid_request",
                      "message": "use max_completion_tokens"}})


def _make_service(tmp_path, cfg: Config, registry: Registry):
    pool = PoolManager(cfg.pool, [Account(id="a1", access_token="t")])
    tokens = TokenManager(cfg, pool)
    store = Store(str(tmp_path / "test.db"))
    capture = JsonlCapture(str(tmp_path / "logs"), enabled=False)
    service = ChatService(cfg, pool, tokens, StubClient(), registry, store, capture)
    return service, pool, tokens, store


@pytest.mark.asyncio
async def test_unknown_model_probe_learns_variant(tmp_path):
    cfg = Config()
    cfg.models.probe_unknown = True
    registry = Registry()
    service, pool, tokens, store = _make_service(tmp_path, cfg, registry)

    payload = {"model": "newvendor/model-x",
               "messages": [{"role": "user", "content": "hi"}]}
    account, headers, body, resp, variant_used = await service._open(
        payload, "default")

    assert variant_used == "openai-nextgen"
    assert "max_completion_tokens" in body
    # the winner is cached for all future calls
    assert registry.variant_for("newvendor/model-x") == "openai-nextgen"

    await resp.aclose()
    await pool.release(account)
    await tokens.aclose()
    store.close()


@pytest.mark.asyncio
async def test_known_model_never_probes(tmp_path):
    cfg = Config()
    registry = Registry()
    service, pool, tokens, store = _make_service(tmp_path, cfg, registry)

    payload = {"model": "cline-pass/glm-5.2",
               "messages": [{"role": "user", "content": "hi"}]}
    account, headers, body, resp, variant_used = await service._open(
        payload, registry.variant_for("cline-pass/glm-5.2"))

    assert variant_used == "default"
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body

    await resp.aclose()
    await pool.release(account)
    await tokens.aclose()
    store.close()


@pytest.mark.asyncio
async def test_probe_disabled_surfaces_client_error(tmp_path):
    from cline_gateway.service import UpstreamFailure

    cfg = Config()
    cfg.models.probe_unknown = False
    registry = Registry()
    service, pool, tokens, store = _make_service(tmp_path, cfg, registry)

    payload = {"model": "newvendor/model-x",
               "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(UpstreamFailure):
        await service._open(payload, "default")

    await tokens.aclose()
    store.close()