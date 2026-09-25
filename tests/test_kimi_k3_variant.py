"""Kimi K3 support — the fourth upstream body variant.

Captured 2026-09-19 from the official Cline desktop client (5 calls,
`cline-free/kimi-k3`, all 200):

    keys      : messages, model, stream, stream_options, tool_choice, tools
                + reasoning control
    token cap : NEITHER max_tokens NOR max_completion_tokens
    thinking  : "reasoning_effort": "medium" | "high" | "xhigh"
                or {"reasoning": {"enabled": false}} when off
    response  : deltas carry `reasoning` + `reasoning_details`; upstream id is
                vmc/fireworks-cline-k3-contributor-fallbacks
"""

from __future__ import annotations

import pytest

from cline_gateway.registry import Registry, model_lane
from cline_gateway.upstream import VARIANTS, build_upstream_body


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_kimi_k3_resolves_to_the_reasoning_variant():
    reg = Registry()
    assert reg.variant_for("cline-free/kimi-k3") == "reasoning"
    assert reg.knows("cline-free/kimi-k3") is True     # no probe needed
    # same family on the other lanes uses the same shape
    assert reg.variant_for("cline-pass/kimi-k3") == "reasoning"
    assert reg.variant_for("cline-cloud/kimi-k3") == "reasoning"
    # pattern rule catches future spellings
    assert reg.variant_for("vendor/kimi-k3-turbo") == "reasoning"


def test_kimi_k3_is_a_free_lane_model():
    assert Registry.is_free("cline-free/kimi-k3") is True
    assert model_lane("cline-free/kimi-k3") == "free"
    assert model_lane("cline-pass/kimi-k3") == "plan"
    assert model_lane("cline-cloud/kimi-k3") == "plan"


def test_removed_free_kimi_is_not_advertised_in_the_catalogue():
    ids = {m["id"] for m in Registry().catalogue()}
    assert "cline-free/kimi-k3" not in ids
    assert "cline-pass/kimi-k3" in ids
    assert "cline-cloud/kimi-k3" in ids


def test_reasoning_variant_is_registered():
    assert "reasoning" in VARIANTS


# --------------------------------------------------------------------------- #
# request body shape
# --------------------------------------------------------------------------- #


def _payload(**extra):
    base = {"model": "cline-free/kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]}
    base.update(extra)
    return base


def test_reasoning_variant_omits_every_token_limit_key():
    body = build_upstream_body(_payload(), "reasoning")
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    # the captured defaults are still there
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tool_choice"] == "auto"


def test_reasoning_off_by_default_matches_the_capture():
    body = build_upstream_body(_payload(), "reasoning")
    assert body["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_observed_effort_values_pass_through(effort):
    body = build_upstream_body(_payload(reasoning_effort=effort), "reasoning")
    assert body["reasoning_effort"] == effort
    assert "reasoning" not in body


def test_low_effort_is_mapped_up_to_medium():
    # "low" was never observed for this family and the client's floor is
    # "medium"; sending an unproven enum risks a 400
    body = build_upstream_body(_payload(reasoning_effort="low"), "reasoning")
    assert body["reasoning_effort"] == "medium"


def test_explicit_reasoning_object_passes_through():
    body = build_upstream_body(
        _payload(reasoning={"enabled": True, "effort": "high"}), "reasoning")
    assert body["reasoning"] == {"enabled": True, "effort": "high"}
    assert "reasoning_effort" not in body


def test_other_variants_still_carry_a_token_limit():
    # regression guard: the shared builder must not have changed shape for the
    # three previously captured variants
    default = build_upstream_body(_payload(), "default")
    assert default["max_tokens"] == 32000
    assert default["reasoning_effort"] == "low"

    nextgen = build_upstream_body(_payload(), "openai-nextgen")
    assert nextgen["max_completion_tokens"] == 32000
    assert "max_tokens" not in nextgen

    anthropic = build_upstream_body(_payload(), "anthropic")
    assert anthropic["max_tokens"] == 32000
    assert anthropic["cache_control"] == {"type": "ephemeral"}


# --------------------------------------------------------------------------- #
# aggregation of streamed reasoning
# --------------------------------------------------------------------------- #


def test_aggregate_collects_streamed_reasoning():
    from cline_gateway.service import aggregate_openai_stream

    sse = (
        'data: {"id":"gen_1","object":"chat.completion.chunk","created":1,'
        '"model":"vmc/fireworks-cline-k3-contributor-fallbacks","choices":'
        '[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"gen_1","choices":[{"index":0,'
        '"delta":{"reasoning":"Let me think"},"finish_reason":null}]}\n\n'
        'data: {"id":"gen_1","choices":[{"index":0,'
        '"delta":{"reasoning":" about it."},"finish_reason":null}]}\n\n'
        'data: {"id":"gen_1","choices":[{"index":0,'
        '"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"gen_1","choices":[{"index":0,"delta":{},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":6000,'
        '"completion_tokens":216,"total_tokens":6216}}\n\n'
        "data: [DONE]\n\n"
    )
    out = aggregate_openai_stream(sse, "cline-free/kimi-k3", "fallback")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Hello"
    assert msg["reasoning"] == "Let me think about it."
    assert out["usage"]["total_tokens"] == 6216
    # the upstream (resolved) model id is what the client sees
    assert out["model"] == "vmc/fireworks-cline-k3-contributor-fallbacks"


def test_aggregate_without_reasoning_has_no_field():
    from cline_gateway.service import aggregate_openai_stream

    sse = ('data: {"id":"g","choices":[{"index":0,"delta":{"content":"hi"},'
           '"finish_reason":"stop"}]}\n\n')
    out = aggregate_openai_stream(sse, "m", "fallback")
    assert "reasoning" not in out["choices"][0]["message"]
